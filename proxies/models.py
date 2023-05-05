import torch.nn as nn
import torch
from phast.embedding import PhysEmbedding
from torch_scatter import scatter
from torch_geometric.nn.dense import DenseGATConv, DenseGCNConv
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.utils import to_dense_batch


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
                input_dim = conv_hidden_channels
            if conv_type == "gat":
                gnn_layers.append(
                    DenseGATConv(
                        input_dim,
                        conv_hidden_channels,
                        head=heads,
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
        count_atoms_per_graph = torch.unique(batch_mask, return_counts=True)[1]
        N_max = max(count_atoms_per_graph)
        batch_size = max(batch_mask).item() + 1
        adj = torch.zeros(batch_size, N_max, N_max)
        for i in range(batch_size):
            adj[i, : count_atoms_per_graph[i], : count_atoms_per_graph[i]] = 1
            adj[i].fill_diagonal_(0)  # optional
        adj = adj.to(device=x.device)
        x = to_dense_batch(x, batch_mask)[0]

        for layer in self.gnn_layers:
            x = layer(x, adj)
            x = self.act(x)
        x = torch.sum(x, dim=1)
        return x


class ProxyMLP(nn.Module):
    def __init__(self, in_feat, num_layers, hidden_channels, cat=True):
        super(ProxyMLP, self).__init__()
        self.concat = cat
        self.hidden_act = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(p=0.5)
        self.final_act = nn.Identity()  # nn.Tanh()
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

    def forward(self, x):
        if self.concat:
            x[1] = x[1].unsqueeze(dim=-1)
            x = torch.cat(x, dim=-1)
        for i, layer in enumerate(self.nn_layers):
            x = layer(x)
            if i == len(self.nn_layers) - 1:
                x = self.final_act(x)
            if i % 2 == 1:
                x = self.hidden_act(x)
                x = self.dropout(x)

        return x


class ProxyEmbeddingModel(nn.Module):
    def __init__(
        self,
        pred_num_layers: int,
        pred_hidden_channels: int,
        comp_size: int,
        comp_num_layers: int,
        comp_hidden_channels: int,
        comp_phys_embeds: int,
        lat_size: int,
        lat_num_layers: int,
        lat_hidden_channels: int,
        sg_emb_size: int,
        alphabet: list = [],
    ):
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
        self.sg_emb = nn.Embedding(230, sg_emb_size)
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

    def forward(self, x):
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
            )
            batch_mask = torch.repeat_interleave(
                torch.arange(comp_x.shape[0]).to(comp_x.device),
                comp_x.sum(dim=1).to(torch.int32),
            )
            comp_x = self.phys_emb(self.alphabet[z].to(torch.int32))
            # comp_x = self.phys_emb(z)
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
        super().__init__()
        # Encoding blocks
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
        self.sg_emb = nn.Embedding(230, sg_emb_size)
        self.lat_emb_mlp = mlp_from_layers(
            lat_num_layers, lat_hidden_channels, lat_size
        )
        # Prediction
        self.pred_inp_size = comp_hidden_channels + sg_emb_size + lat_hidden_channels
        self.prediction_head = ProxyMLP(
            self.pred_inp_size, pred_num_layers, pred_hidden_channels, False
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

    def forward(self, x):
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
        # comp_x = self.phys_emb(z)
        comp_x = self.phys_emb(self.alphabet[z].to(torch.int32))

        # Add space group and lattice embeddings to node attributes
        if self.add_to_node:
            comp_x = torch.cat((comp_x, lat_x[batch_mask], sg_x[batch_mask]), dim=-1)

        # Apply a GNN to the node attributes
        comp_x = self.conv(comp_x, batch_mask)

        # Concatenate and predict
        x = torch.cat((comp_x, sg_x, lat_x), dim=-1)
        return self.prediction_head(x)
