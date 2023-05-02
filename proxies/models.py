import torch.nn as nn
import torch
from phast.embedding import PhysEmbedding
from torch_scatter import scatter


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)


def make_model(config):
    if config["config"].startswith("mlp-"):
        model = ProxyMLP(config["model"]["input_len"], config["model"]["hidden_layers"])
        model.apply(weights_init)
        return model
    elif config["config"].startswith("physmlp-"):
        model = ProxyEmbeddingModel(
            comp_emb_layers=config["model"]["comp_emb_layers"],
            comp_phys_embeds=config["model"]["comp_phys_embeds"],
            sg_emb_size=config["model"]["sg_emb_size"],
            lat_emb_layers=config["model"]["lat_emb_layers"],
            prediction_layers=config["model"]["hidden_layers"],
            alphabet=config["alphabet"],
        )
        return model
    else:
        raise ValueError(f"Unknown model config: {config['config']}")


def mlp_from_layers(layers, act=None, norm=True):
    nn_layers = []
    for i in range(len(layers)):
        try:
            nn_layers.append(nn.Linear(layers[i], layers[i + 1]))
            if norm:
                nn_layers.append(nn.BatchNorm1d(layers[i + 1]))
            nn_layers.append(nn.LeakyReLU(True) if act is None else act)
        except IndexError:
            pass
    nn_layers = nn.Sequential(*nn_layers)
    return nn_layers


class ProxyMLP(nn.Module):
    def __init__(self, in_feat, hidden_layers, cat=True):
        super(ProxyMLP, self).__init__()
        self.concat = cat
        self.hidden_act = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(p=0.5)
        self.final_act = nn.Identity()  # nn.Tanh()
        # Model archi
        self.nn_layers = nn.ModuleList()
        for i in range(len(hidden_layers)):
            if i == 0:
                self.nn_layers.append(nn.Linear(in_feat, hidden_layers[i]))
            else:
                self.nn_layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            self.nn_layers.append(nn.BatchNorm1d(hidden_layers[i]))
        self.nn_layers.append(nn.Linear(hidden_layers[-1], 1))

    def forward(self, x):
        if self.concat:
            x = torch.cat([x[1], x[2], x[0]], dim=-1)
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
        comp_emb_layers: list,
        comp_phys_embeds: dict,
        sg_emb_size: int,
        lat_emb_layers: list,
        prediction_layers: list,
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
                final_proj_size=comp_emb_layers[-1],
            )
        else:
            self.comp_emb_mlp = mlp_from_layers(comp_emb_layers)
        self.sg_emb = nn.Embedding(230, sg_emb_size)
        self.lat_emb_mlp = mlp_from_layers(lat_emb_layers)
        self.pred_inp_size = comp_emb_layers[-1] + sg_emb_size + lat_emb_layers[-1]
        self.prediction_head = ProxyMLP(self.pred_inp_size, prediction_layers, False)
        if not alphabet:
            self._alphabet = torch.Tensor(list(range(comp_emb_layers[0])))
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
                (comp_x[idx[:, 0], idx[:, 1].to(torch.int32)]),
                dim=0,
            )
            batch_mask = torch.repeat_interleave(
                torch.arange(comp_x.shape[0]).to(comp_x.device),
                comp_x.sum(dim=1).to(torch.int32),
            )
            comp_x = self.phys_emb(self.alphabet[z].to(torch.int32))
            comp_x = scatter(comp_x, batch_mask, dim=0, reduce="add")
        else:
            comp_x = self.comp_emb_mlp(comp_x)

        # Process the space group
        sg_x = self.sg_emb(sg_x).squeeze(1)

        # Process the lattice
        lat_x = self.lat_emb_mlp(lat_x)

        # Concatenate and predict
        x = torch.cat((comp_x, sg_x, lat_x), dim=-1)
        return self.prediction_head(x)
