import torch.nn as nn
import torch
from phast.embedding import PhysEmbedding
from torch_scatter import scatter
from torch_geometric.nn.dense import DenseGATConv, DenseGCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import models as graph_nn
from torch_geometric.utils import to_dense_batch
from faenet.model import FAENet
from faenet.fa_forward import model_forward
from dave.utils.gnn import GaussianSmearing


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)


def make_model(config):
    # MLP
    if config["config"].startswith("mlp-"):
        model = ProxyMLP(
            in_feat=config["model"]["input_len"],
            num_layers=config["model"]["num_layers"],
            hidden_channels=config["model"]["hidden_channels"],
            concat=config["model"]["concat"],
        )
        model.apply(weights_init)
        return model
    # MLP with physical embeddings to model composition
    elif config["config"].startswith("physmlp-"):
        model = ProxyEmbeddingModel(
            pred_num_layers=config["model"]["num_layers"],
            pred_hidden_channels=config["model"]["hidden_channels"],
            comp_size=config["comp_size"],
            comp_num_layers=config["model"]["comp_num_layers"],
            comp_hidden_channels=config["model"]["comp_hidden_channels"],
            lat_size=config["lat_size"],
            lat_num_layers=config["model"]["lat_num_layers"],
            lat_hidden_channels=config["model"]["lat_hidden_channels"],
            sg_emb_size=config["model"]["sg_emb_size"],
            comp_phys_embeds=config["model"]["comp_phys_embeds"],
            alphabet=config["alphabet"],
        )
        return model
    # Graph model without 3D pos.
    elif config["config"].startswith("graph-"):
        model = ProxyGraphModel(
            pred_num_layers=config["model"]["num_layers"],
            pred_hidden_channels=config["model"]["hidden_channels"],
            comp_size=config["comp_size"],
            comp_num_layers=config["model"]["comp_num_layers"],
            comp_hidden_channels=config["model"]["comp_hidden_channels"],
            comp_phys_embeds=config["model"]["comp_phys_embeds"],
            lat_size=config["lat_size"],
            lat_num_layers=config["model"]["lat_num_layers"],
            lat_hidden_channels=config["model"]["lat_hidden_channels"],
            sg_emb_size=config["model"]["sg_emb_size"],
            alphabet=config["alphabet"],
            conv=config["model"]["conv"],
        )
        return model
    elif config["config"].startswith("pyxtal"):
        model = Pyxtal_FAENet(config["frame_averaging"], **config["model"])
        return model
    elif config["config"].startswith("fae-"):
        model = ArchFAE()
        return model
    elif config["config"].startswith("faecry-"):
        model = FAENet()
        return model
    elif config["config"].startswith("sch-"):
        model = BaSch(
            config["model"]["hidden_channels"],
            config["model"]["num_filters"],
            config["model"]["num_interactions"],
            config["model"]["readout"],
        )
        return model
    else:
        raise ValueError(f"Unknown model config: {config['config']}")


def mlp_from_layers(num_layers, hidden_channels, input_size=None, act=None, norm=True):
    nn_layers = []
    for i in range(num_layers):
        if i == 0:
            nn_layers.append(nn.Linear(input_size, hidden_channels))
        else:
            nn_layers.append(nn.Linear(hidden_channels, hidden_channels))
        if norm:
            nn_layers.append(nn.BatchNorm1d(hidden_channels))
        nn_layers.append(nn.LeakyReLU(True) if act is None else act)
    nn_layers = nn.Sequential(*nn_layers)
    return nn_layers


class GNNBlock(nn.Module):
    """GNN block which updates atom representations"""

    def __init__(
        self,
        input_dim,
        conv_num_layers,
        conv_hidden_channels,
        conv_type="gat",
        heads=3,
        concat=True,
        dropout=0.0,
        norm=False,
        act=None,
    ):
        super(GNNBlock, self).__init__()

        gnn_layers = nn.ModuleList()

        for i in range(conv_num_layers):
            if i > 0:
                input_dim = conv_hidden_channels * heads
            if conv_type == "gat":
                gnn_layers.append(
                    DenseGATConv(
                        input_dim,
                        conv_hidden_channels,
                        heads=heads,
                        concat=concat,
                        dropout=dropout,
                    )
                )
            else:
                gnn_layers.append(DenseGCNConv(input_dim, conv_hidden_channels))
            # if norm:
            #     gnn_layers.append(GraphNorm(conv_layers[i + 1]))
        self.act = nn.LeakyReLU(True) if act is None else act
        self.gnn_layers = gnn_layers

    def forward(self, x, batch_mask):
        # Create a complete graph adjacency matrix for each batch: dim [B, N, N]
        # Feature matrix has dimension [B, N, F]
        count_atoms_per_graph = torch.unique(batch_mask, return_counts=True)[1]
        N_max = max(count_atoms_per_graph)
        batch_size = max(batch_mask).item() + 1
        adj = torch.zeros(batch_size, N_max, N_max)
        for i in range(batch_size):
            adj[i, : count_atoms_per_graph[i], : count_atoms_per_graph[i]] = 1
            adj[i].fill_diagonal_(0)  # optional
        adj = adj.to(device=x.device)
        x = to_dense_batch(x, batch_mask)[0]

        for l, layer in enumerate(self.gnn_layers):
            x = layer(x, adj)
            x = self.act(x)
        x = torch.sum(x, dim=1)
        return x


class ProxyMLP(nn.Module):
    """Proxy model for the prediction of the formation energy of a crystal structure.
    MLP of composition (and space group and lattice parameters).

    Args:
        in_feat (int): number of input features (composition size)
        num_layers (int): number of MLP layers
        hidden_channels (int): number of hidden channels
        concat (bool, optional): concatenate space group
            and lattice to the composition. Defaults to True.
    """

    def __init__(self, in_feat, num_layers, hidden_channels, concat=True):
        super(ProxyMLP, self).__init__()
        self.concat = concat
        self.hidden_act = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(p=0.5)
        self.final_act = nn.Identity()  # nn.Tanh()
        if self.concat:
            in_feat += 7  # space group + lattice params
        # Model archi
        self.nn_layers = nn.ModuleList()
        for i in range(num_layers - 1):
            if i == 0:
                self.nn_layers.append(nn.Linear(in_feat, hidden_channels))
            else:
                self.nn_layers.append(nn.Linear(hidden_channels, hidden_channels))
            self.nn_layers.append(nn.BatchNorm1d(hidden_channels))
        if num_layers < 1:
            hidden_channels = in_feat
        self.nn_layers.append(nn.Linear(hidden_channels, 1))

    def forward(self, x, batch=None):
        if self.concat:  # composition + sg + lattice
            x[1] = x[1].unsqueeze(dim=-1)
            x = torch.cat(x, dim=-1)
        elif isinstance(x, list):  # keep composition only
            x = x[0]
        # MLP
        for i, layer in enumerate(self.nn_layers):
            x = layer(x)
            if i == len(self.nn_layers) - 1:
                x = self.final_act(x)
            if i % 2 == 1:
                x = self.hidden_act(x)
                x = self.dropout(x)
        return x


class ProxyEmbeddingModel(nn.Module):
    """
    Proxy model for the prediction of the formation energy of a crystal structure.
    MLP of composition, space group and lattice parameters,
    where composition is modelled using physics aware embeddings.
    """

    def __init__(
        self,
        pred_num_layers: int,
        pred_hidden_channels: int,
        comp_size: int,
        comp_num_layers: int,
        comp_hidden_channels: int,
        comp_phys_embeds: dict,
        lat_size: int,
        lat_num_layers: int,
        lat_hidden_channels: int,
        sg_emb_size: int,
        alphabet: list = [],
    ):
        """
        Args:
            pred_num_layers (int): number of MLP layers for the prediction head
            pred_hidden_channels (int): number of hidden channels for the prediction head
            comp_size (int): total number of chemical elements
            comp_num_layers (int): number of MLP layers to model composition
            comp_hidden_channels (int): number of hidden channels to model composition
            comp_phys_embeds (dict): dictionary with various parameters for the physics aware embeddings
                of the composition
                    - use (bool): whether to use physics aware embeddings
                    - z_emb_size: 32
                    - period_emb_size: 16
                    - group_emb_size: 16
                    - properties_proj_size: 32
                    - n_elements: 90
            lat_size (int): number of lattice parameters
            lat_num_layers (int): number of MLP layers for the lattice parameters
            lat_hidden_channels (int): number of hidden channels for the lattice parameters
            sg_emb_size (int): size of the learned space group embedding
            alphabet (list, optional): _description_. Defaults to [].
        """
        super().__init__()
        self.use_comp_phys_embeds = comp_phys_embeds["use"]
        if self.use_comp_phys_embeds:
            self.phys_emb = PhysEmbedding(
                z_emb_size=comp_phys_embeds["z_emb_size"],
                period_emb_size=comp_phys_embeds["period_emb_size"],
                group_emb_size=comp_phys_embeds["group_emb_size"],
                properties_proj_size=comp_phys_embeds["properties_proj_size"],
                n_elements=max(alphabet) + 1,
                final_proj_size=comp_hidden_channels,
            )
        else:
            self.comp_emb_mlp = mlp_from_layers(
                comp_num_layers, comp_hidden_channels, comp_size
            )
        self.sg_emb = nn.Embedding(231, sg_emb_size)
        self.lat_emb_mlp = mlp_from_layers(
            lat_num_layers, lat_hidden_channels, lat_size
        )
        self.pred_inp_size = comp_hidden_channels + sg_emb_size + lat_hidden_channels
        self.prediction_head = ProxyMLP(
            self.pred_inp_size, pred_num_layers, pred_hidden_channels, False
        )
        if not alphabet:
            self._alphabet = torch.Tensor(list(range(comp_size)))
        else:
            self._alphabet = torch.Tensor(alphabet)
        self.register_buffer("alphabet", self._alphabet)

    def forward(self, x, batch=None):
        comp_x, sg_x, lat_x = x
        # comp_x -> batch_size x n_elements=89
        # sg_x -> batch_size, int
        # lat_x -> batch_size x 6

        # Process the composition
        if self.use_comp_phys_embeds:
            idx = torch.nonzero(comp_x)
            z = torch.repeat_interleave(
                idx[:, 1],
                (comp_x[idx[:, 0], idx[:, 1]]),
                dim=0,
            ).long()
            batch_mask = torch.repeat_interleave(
                torch.arange(comp_x.shape[0]).to(comp_x.device),
                comp_x.sum(dim=1).long(),
            )
            comp_x = self.phys_emb(z).to(torch.float32)
            # comp_x = self.phys_emb(self.alphabet[z].to(torch.int32))
            comp_x = scatter(comp_x, batch_mask, dim=0, reduce="mean")
        else:
            comp_x = self.comp_emb_mlp(comp_x)

        # Process the space group
        sg_x = self.sg_emb(sg_x).squeeze(1)

        # Process the lattice
        lat_x = self.lat_emb_mlp(lat_x)

        # TODO: add a counter for the number of atoms in the unit cell

        # Concatenate and predict
        x = torch.cat((comp_x, sg_x, lat_x), dim=-1)
        return self.prediction_head(x)


class ProxyGraphModel(nn.Module):
    """
    Proxy model for the prediction of the formation energy of a crystal structure.
    GNN model to embed the composition, MLP for space group and lattice params.
    """

    def __init__(
        self,
        pred_num_layers: int,
        pred_hidden_channels: int,
        comp_num_layers: int,
        comp_size: int,
        comp_hidden_channels: int,
        sg_emb_size: int,
        lat_size: int,
        lat_hidden_channels: int,
        lat_num_layers: int,
        comp_phys_embeds: int,
        alphabet: list,
        conv: dict,
    ):
        """
        Args:
            pred_num_layers (int): number of MLP layers for the prediction head
            pred_hidden_channels (int): number of hidden channels for the prediction head
            comp_size (int): total number of chemical elements
            comp_num_layers (int): number of MLP layers to model composition
            comp_hidden_channels (int): number of hidden channels to model composition
            comp_phys_embeds (dict): dictionary with various parameters for the physics aware embeddings
                of the composition
                    - z_emb_size: 32
                    - period_emb_size: 16
                    - group_emb_size: 16
                    - properties_proj_size: 32
                    - n_elements: 90
            lat_size (int): number of lattice parameters
            lat_num_layers (int): number of MLP layers for the lattice parameters
            lat_hidden_channels (int): number of hidden channels for the lattice parameters
            sg_emb_size (int): size of the learned space group embedding
            alphabet (list, optional): _description_. Defaults to [].
            conv (dict): dictionary of attributes for the graph convolution layer
                    - num_layers: 2
                    - hidden_channels: 64
                    - type: "gcn"
                    - heads: 4
                    - dropout: 0.5
                    - concat: True
                    - add_to_node: True
        """
        super().__init__()
        # Encoding blocks
        self.phys_emb = PhysEmbedding(
            z_emb_size=comp_phys_embeds["z_emb_size"],
            period_emb_size=comp_phys_embeds["period_emb_size"],
            group_emb_size=comp_phys_embeds["group_emb_size"],
            properties_proj_size=comp_phys_embeds["properties_proj_size"],
            n_elements=max(alphabet) + 1,
            final_proj_size=comp_hidden_channels,
        )

        self.sg_emb = nn.Embedding(231, sg_emb_size)
        self.lat_emb_mlp = mlp_from_layers(
            lat_num_layers, lat_hidden_channels, lat_size
        )
        # Prediction
        self.pred_inp_size = (
            conv["hidden_channels"] * conv.get("heads", 1)
            + sg_emb_size
            + lat_hidden_channels
        )
        self.prediction_head = ProxyMLP(
            self.pred_inp_size,
            pred_num_layers,
            pred_hidden_channels,
            False,
        )
        # Interaction blocks (GNN)
        self.add_to_node = conv["add_to_node"]
        if self.add_to_node:
            self.conv = GNNBlock(
                self.pred_inp_size,
                conv["num_layers"],
                conv["hidden_channels"],
                conv["type"],
                conv["heads"],
                conv["concat"],
                conv["dropout"],
            )
        else:
            self.conv = GNNBlock(
                comp_hidden_channels,
                conv["num_layers"],
                conv["hidden_channels"],
                conv["type"],
                conv["heads"],
                conv["concat"],
                conv["dropout"],
            )
        # Deal with atomic numbers
        if not alphabet:
            self._alphabet = torch.Tensor(list(range(comp_size)))
        else:
            self._alphabet = torch.Tensor(alphabet)
        self.register_buffer("alphabet", self._alphabet)

    def forward(self, x, batch=None):
        # Process the space group
        sg_x = x[1].long()
        sg_x = self.sg_emb(sg_x).squeeze(1)

        # Process the lattice
        lat_x = self.lat_emb_mlp(x[2])

        # Process the composition to create node attributes
        idx = torch.nonzero(x[0])
        # Transform atomic numbers to sparse indices
        z = torch.repeat_interleave(
            idx[:, 1], (x[0][idx[:, 0], idx[:, 1]]).to(torch.int32), dim=0
        )
        # Create sparse mask representing batch
        batch_mask = torch.repeat_interleave(
            torch.arange(x[0].shape[0]).to(x[0].device),
            x[0].sum(dim=1).to(torch.int32),
        )
        # Derive physics aware embeddings
        comp_x = self.phys_emb(z).to(torch.float32)
        # comp_x = self.phys_emb(self.alphabet[z].to(torch.int32))

        # Add space group and lattice embeddings to node attributes
        if self.add_to_node:
            comp_x = torch.cat((comp_x, lat_x[batch_mask], sg_x[batch_mask]), dim=-1)

        # Apply a GNN to the node attributes
        comp_x = self.conv(comp_x, batch_mask)

        # Concatenate and predict
        x = torch.cat((comp_x, sg_x, lat_x), dim=-1)
        return self.prediction_head(x)


class GLFAENet(nn.Module):
    # PROTOTYPE: to be updated
    def __init__(self, config):
        super().__init__()
        self.fae = FAENet(config)

    def forward(self, x):
        x = self.fae(x)
        return model_forward(x)


class ArchFAE(nn.Module):
    """FAENet model"""

    def __init__(
        self,
        comp_size: int,
        comp_num_layers: int,
        comp_hidden_channels: int,
        comp_phys_embeds: int,
    ):
        super().__init__()
        self.base_fae = FAENet(tag_hidden_channels=0)
        hidden_channels = self.base_fae.hidden_channels
        phys_hidden_channels = self.base_fae.phys_hidden_channels
        pg_hidden_channels = self.base_fae.pg_hidden_channels
        self.replace_emb = nn.Embedding(
            comp_size, hidden_channels - phys_hidden_channels - 2 * pg_hidden_channels
        )
        self.base_fae.emb = self.replace_emb

    def forward(self, data, batch=None):
        node_level_preds = self.base_fae.energy_forward(data)["energy"]
        return global_mean_pool(node_level_preds, batch)


class BaSch(nn.Module):
    """SchNet model"""

    def __init__(self, hidden_channels, num_filters, num_interactions, readout):
        super().__init__()
        self.schnet = graph_nn.SchNet(
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_interactions=num_interactions,
            num_gaussians=60,
            cutoff=6.0,
            max_num_neighbors=50,
            readout=readout,
        )

    def forward(self, x, batch_idx):
        z = x.atomic_numbers.int()
        atom_pos = x.pos
        return self.schnet(z, atom_pos, x.batch)


class Pyxtal_FAENet(nn.Module):
    """FAENet model applied on Pyxtal data structures"""

    def __init__(self, frame_averaging, **kwargs):
        super().__init__()
        self.faenet = FAENet(**kwargs)
        self.frame_averaging = frame_averaging
        # TODO: REMOVE this two when FAENet package is updated
        self.faenet.embed_block.emb = nn.Embedding(
            100, kwargs["hidden_channels"] - kwargs["phys_hidden_channels"] - 2 * kwargs["pg_hidden_channels"]
        )
        # self.faenet.distance_expansion = GaussianSmearing(0.0, self.faenet.cutoff, self.faenet.num_gaussians)
        self.faenet.forward = self.faenet_forward
    
    def faenet_forward(self, data, mode="train", preproc=True):
        """Main Forward pass.

        Args:
            data (Data): input data object, with 3D atom positions (pos)
            mode (str): train or inference mode
            preproc (bool): Whether to preprocess (pbc, cutoff graph)
                the input graph or point cloud. Default: True.

        Returns:
            (dict): predicted energy, forces and final atomic hidden states
        """
        # energy gradient w.r.t. positions will be computed
        if mode == "train" or self.faenet.regress_forces == "from_energy":
            data.pos.requires_grad_(True)

        # predict energy
        preds = self.faenet.energy_forward(data, preproc)

        # Predict atom positions 
        preds["forces"] = self.faenet.forces_forward(preds)

        return preds

    def forward(self, data, batch_idx=None):
        """data: data.Batch batch of graphs with attributes:
        - pos: original atom positions
        - batch: indices (to which graph in batch each atom belongs to)
        - fa_pos, fa_cell, fa_rot: frame averaged positions, cell and rotation matrices
        """
        out = model_forward(
            data,
            self.faenet,
            frame_averaging=self.frame_averaging,
            mode="train",
            crystal_task=False,
        )
        return out["forces"]
