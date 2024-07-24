from pathlib import Path

import torch
import torch.nn as nn
from phast.embedding import PhysEmbedding
from torch_geometric.nn.dense import DenseGATConv, DenseGCNConv
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter
from yaml import safe_load

from dave.utils.symmetries import all_symmetry_vectors


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)


def nan_grad_to_zero(grad):
    grad[grad != grad] = 0
    return grad


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
            sg_encoder_config=config["model"].get("sg_encoder", {}),
            wyckoff_config=config["model"].get("wyck_encoder", {}),
        )
        model.apply(weights_init)
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
        # x, _ = x
        if self.concat:
            x[1] = x[1].unsqueeze(dim=-1)
            x = torch.cat(x[:, :3], dim=-1)
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
        sg_encoder_config: dict = {},
        wyckoff_config: dict = {},
    ):
        super().__init__()
        self.use_comp_phys_embeds = comp_phys_embeds["use"]

        # Physical embeddings for atom types
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

        # Symmetry-based embeddings for space groups
        if sg_encoder_config.get("use"):
            if sg_encoder_config.get("sg_yaml"):
                base = sg_encoder_config["sg_yaml"]
                pointsymms = safe_load(open(str(Path(base) / "point_symmetries.yaml")))
                cryslatsys = safe_load(
                    open(str(Path(base) / "crystal_lattice_systems.yaml"))
                )
            else:
                try:
                    from crystallograpy import pointsymms, cryslatsys
                except ImportError:
                    print(
                        "Please configure crystallograpy repo or pass the path to the relevant YAML files"
                    )
                    raise ImportError
            # sg_encoder_config["sg_to_ps_dict"] = pointsymms
            # sg_encoder_config["sg_to_cls_dict"] = cryslatsys
            self.sg_emb = SpaceGroupEncoder(
                **sg_encoder_config, sg_to_ps_dict=pointsymms, sg_to_cls_dict=cryslatsys
            )
            sg_emb_size = self.sg_emb.output_size
        else:
            # basic space groups embedding
            self.sg_emb = nn.Embedding(231, sg_emb_size)

        if wyckoff_config.get("use"):
            self.use_wyck_embeds = wyckoff_config["use"]
        else:
            self.use_wyck_embeds = False
            wyck_emb_size = 0

        if self.use_wyck_embeds:
            wyck_emb_size = wyckoff_config["wyck_embed_size"]
            base = wyckoff_config.get("wyck_lm_yaml")
            if base:
                base = Path(base)
                wyck_dix = safe_load(open(str(base / "wyckoff_lm_embeddings.yaml")))
            else:
                wyck_dix = {}
            self.wyck_emb = WyckoffEncoder(wyck_emb_size, wyck_dix)

        # lattice parameters MLP
        self.lat_emb_mlp = mlp_from_layers(
            lat_num_layers, lat_hidden_channels, lat_size
        )

        # compute full embedding size
        self.pred_inp_size = (
            comp_hidden_channels + wyck_emb_size + sg_emb_size + lat_hidden_channels
        )

        # output MLP
        self.prediction_head = ProxyMLP(
            self.pred_inp_size, pred_num_layers, pred_hidden_channels, False
        )

        # Alphabet mapping {index: atomic_number}
        if not alphabet:
            self._alphabet = torch.Tensor(list(range(comp_size)))
        else:
            self._alphabet = torch.Tensor(alphabet)
        self.register_buffer("alphabet", self._alphabet)

    def forward(self, x):
        comp_x, sg_x, lat_x, wy_x = x
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
            # comp_x = self.phys_emb(self.alphabet[z].to(torch.int32))
            comp_x = self.phys_emb(z).to(torch.int32)
            comp_h = scatter(comp_x, batch_mask, dim=0, reduce="mean")
        else:
            comp_h = self.comp_emb_mlp(comp_x)

        if self.use_wyck_embeds:
            wy_h = self.wyck_emb(wy_x)
            comp_h = torch.cat((comp_h, wy_h), dim=-1)

        # Process the space group
        sg_h = self.sg_emb(sg_x).squeeze(1)

        # Process the lattice
        lat_h = self.lat_emb_mlp(lat_x)

        # TODO: add a counter for the number of atoms in the unit cell

        # Concatenate and predict
        h = torch.cat((comp_h, sg_h, lat_h), dim=-1)
        return self.prediction_head(h)


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


class SpaceGroupEncoder(nn.Module):
    def __init__(
        self,
        symmetries_hiddens=[16],
        symmetries_aggregation="sum",
        space_group_size=16,
        point_symmetry_size=16,
        cl_system_size=16,
        sg_to_ps_dict={},
        sg_to_cls_dict={},
        **kwargs,
    ):
        """
        Space group encoder.

        Args:
            symmetries_hiddens (list, optional): Hidden layers for the symmetry
                vectors. Defaults to [16].
            symmetries_aggregation (str, optional): Aggregation stratey for the
                symmetry vectors. Defaults to "sum".
            space_group_size (int, optional): Space group embedding size.
                Defaults to 16.
            point_symmetry_size (int, optional): Point symmetry embedding size.
                Defaults to 16.
            cl_system_size (int, optional): Crystal lattice system embedding size.
                Defaults to 16.
            kwargs (dict): Additional ignored arguments.
        """
        super().__init__()

        if isinstance(symmetries_hiddens, int):
            symmetries_hiddens = [symmetries_hiddens]
        elif isinstance(symmetries_hiddens, str):
            symmetries_hiddens = [int(h.strip()) for h in symmetries_hiddens.split(",")]

        self.symmetries_hiddens = symmetries_hiddens if symmetries_hiddens else []
        self.space_group_size = int(space_group_size) if space_group_size else 0
        self.point_symmetry_size = (
            int(point_symmetry_size) if point_symmetry_size else 0
        )
        self.cl_system_size = int(cl_system_size) if cl_system_size else 0
        self.symmetries_aggregation = symmetries_aggregation

        self.symmetries_encoder = None
        if symmetries_hiddens:
            self.symmetries_encoder = []
            sizes = [7] + symmetries_hiddens
            for i, output_dim in enumerate(symmetries_hiddens):
                input_dim = sizes[i]
                self.symmetries_encoder.append(nn.Linear(input_dim, output_dim))
                if i < len(symmetries_hiddens) - 1:
                    self.symmetries_encoder.append(nn.SiLU())
            self.symmetries_encoder = nn.Sequential(*self.symmetries_encoder)
            for param in self.symmetries_encoder.parameters():
                # prevent nan grads to 0 with register hook
                param.register_hook(nan_grad_to_zero)

        self.space_group_encoder = (
            nn.Embedding(230 + 1, space_group_size) if space_group_size > 0 else None
        )
        self.point_symmetry_encoder = (
            nn.Embedding(6 + 1, point_symmetry_size)
            if point_symmetry_size > 0
            else None
        )
        self.cl_system_encoder = (
            nn.Embedding(8 + 1, point_symmetry_size)
            if point_symmetry_size > 0
            else None
        )

        # Make tensor that maps space groups to point symmetries
        # sg_to_ps_dict = safe_load(open(FILES["ps"]))
        sg_to_ps_dict = {
            sg: ps for ps, psd in sg_to_ps_dict.items() for sg in psd["space_groups"]
        }
        sg_to_ps = torch.full((max(sg_to_ps_dict.keys()) + 1,), -1, dtype=torch.long)
        for sg, ps in sg_to_ps_dict.items():
            sg_to_ps[sg] = ps
        self.register_buffer("sg_to_ps", sg_to_ps)

        # Make tensor that maps space groups to crystal lattice systems
        # sg_to_cls_dict = safe_load(open(FILES["cls"]))
        sg_to_cls_dict = {
            sg: cls
            for cls, clsd in sg_to_cls_dict.items()
            for sg in clsd["space_groups"]
        }
        sg_to_cls = torch.full((max(sg_to_cls_dict.keys()) + 1,), -1, dtype=torch.long)
        for sg, cls in sg_to_cls_dict.items():
            sg_to_cls[sg] = cls
        self.register_buffer("sg_to_cls", sg_to_cls)

        # Precompute symmetry vectors for all space groups
        # Make 3D tensor with all symmetry vectors for all space groups
        # Padding empty rows groups with NaNs
        print("Precomputing space group symmetries...", end="", flush=True)
        sg_to_sym_vects_dict = {
            space_group: torch.from_numpy(all_symmetry_vectors(space_group)).float()
            for space_group in range(1, 231)
        }
        max_n_vects = max([v.shape[0] for v in sg_to_sym_vects_dict.values()])
        sg_to_sym_vects = torch.full(
            (max(sg_to_sym_vects_dict.keys()) + 1, max_n_vects, 7), float("nan")
        )
        for sg, vects in sg_to_sym_vects_dict.items():
            sg_to_sym_vects[sg, : vects.shape[0]] = vects
        self.register_buffer("sg_to_sym_vects", sg_to_sym_vects)
        print(" ok.")

        self._output_size = 0

    @property
    def output_size(self):
        if not self._output_size:
            size = 0
            if self.symmetries_encoder:
                size += self.symmetries_hiddens[-1]
            size += max(self.space_group_size, 0)
            size += max(self.point_symmetry_size, 0)
            size += max(self.cl_system_size, 0)
            self._output_size = size
        return self._output_size

    def aggregate_symmetry_embeddings(self, embedded_symetry_vectors: torch.Tensor):
        """
        Aggregate symmetry embeddings according to ``self.symmetries_aggregation``.

        Args:
            embedded_symetry_vectors (torch.Tensor): Symmetry embeddings as [N, D]

        Raises:
            ValueError: If the aggregation method is unknown.

        Returns:
            torch.Tensor: Aggregated symmetry embeddings as [D]
        """
        if self.symmetries_aggregation == "sum":
            return torch.nansum(embedded_symetry_vectors, 1)
        elif self.symmetries_aggregation == "mean":
            return torch.nanmean(embedded_symetry_vectors, 1)
        raise ValueError(f"Unknown aggregation method: {self.symmetries_aggregation}")

    def embed_symmetries(self, space_group: int):
        """
        Embed symmetry vectors for a given space group if ``self.symmetries_encoder`` is
        available, then aggregate the embeddings according to
        ``self.symmetries_aggregation``.

        Args:
            space_group (int): Space group number.

        Returns:
            torch.Tensor: Symmetry embeddings as [D] or ``None`` if the encoder is
                not available.
        """
        if self.symmetries_encoder:
            embeddings = self.symmetries_encoder(self.sg_to_sym_vects[space_group])
            return self.aggregate_symmetry_embeddings(embeddings)

    def embed_space_group(self, space_group: int):
        """
        Embed the space group number if ``self.space_group_encoder`` is available.

        Args:
            space_group (int): Space group number.

        Returns:
            torch.Tensor: Space group embedding as [D] or ``None`` if the encoder is
                not available.
        """
        if self.space_group_encoder:
            return self.space_group_encoder(space_group.long())

    def embed_point_symmetry(self, space_group: int):
        """
        Embed the point symmetry if ``self.point_symmetry_encoder`` is available.

        Args:
            space_group (int): Space group number.

        Returns:
            torch.Tensor: Point symmetry embedding as [D] or ``None`` if the encoder is
                not available.
        """
        if self.point_symmetry_encoder:
            return self.point_symmetry_encoder(self.sg_to_ps[space_group])

    def embed_cl_system(self, space_group: int):
        """
        Embed the crystal lattice system if ``self.cl_system_encoder`` is available.

        Args:
            space_group (int): Space group number.

        Returns:
            torch.Tensor: Crystal lattice system embedding as [D] or ``None`` if the
                encoder is not available.
        """
        if self.cl_system_encoder:
            return self.cl_system_encoder(self.sg_to_ps[space_group])

    def forward(self, space_group: int, as_dict=False):
        """
        Embed all symmetries for a given space group and return the embeddings as a
        concatenated tensor.

        Order:
        - Symmetries (as aggregated tensor)
        - Point symmetry
        - Crystal lattice system
        - Space group

        Args:
            space_group (int): Space group number.
            as_dict (bool, optional): Return the embeddings as a dict. Mainly to debug.
                Defaults to False.

        Returns:
            torch.Tensor or list: Embeddings as a concatenated tensor or a list of
                tensors.
        """
        embeddings = {
            "syms": self.embed_symmetries(space_group),
            "ps": self.embed_point_symmetry(space_group),
            "cls": self.embed_cl_system(space_group),
            "sg": self.embed_space_group(space_group),
        }
        if as_dict:
            return embeddings
        return torch.cat([h for h in embeddings.values() if h is not None], -1)


class WyckoffEncoder(nn.Module):
    def __init__(
        self,
        emb_size: int = 64,
        wyckoff_dix: dict = {},
    ):
        super(WyckoffEncoder, self).__init__()
        if wyckoff_dix:
            pretrained = torch.zeros((64, 1))
            for keys, values in wyckoff_dix.items():
                embed = torch.FloatTensor(values["mean"]).unsqueeze(-1)
                pretrained = torch.cat((pretrained, embed), dim=-1)
            pretrained = pretrained.transpose(0, 1)
            self.embedding_layer = nn.Embedding.from_pretrained(pretrained)
        else:
            self.embedding_layer = nn.Embedding(991, emb_size)

    def forward(self, wyck_x):
        wyck_i = wyck_x[:, -1]
        wyck_h = self.embedding_layer(wyck_i)
        return wyck_h.mean(dim=1)
