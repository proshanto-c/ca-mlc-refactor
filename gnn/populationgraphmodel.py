import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.data import HeteroData
    from torch_geometric.nn import GATConv, HeteroConv, SAGEConv
except ImportError as exc:
    raise SystemExit("Missing dependency: torch-geometric. Install it before using this model.") from exc


class PopulationGraphModel(nn.Module):
    """
    Population Hypergraph Neural Network with optional Multiscale (Patch) support.
    """
    def __init__(
        self,
        num_labels: int,
        num_contexts: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        heads: int = 2,
        macro_arch: str = "bipartite",
        initial_biases: list = None,
        multiscale: bool = False
    ) -> None:
        super().__init__()
        
        self.num_labels = num_labels
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.macro_arch = macro_arch
        self.multiscale = multiscale
        
        cnn_out_dim = 512

        if self.multiscale:
            self.patch_encoder = nn.Sequential(
                nn.Linear(cnn_out_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        else:
            self.image_encoder = nn.Sequential(
                nn.Linear(cnn_out_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        
        # --- Context / Label Embeddings ---
        self.context_embed = nn.Embedding(num_contexts, hidden_dim) if num_contexts > 0 else None
        self.label_embed = nn.Embedding(num_labels, hidden_dim)
        
        type_bias_dict = {
            "image": nn.Parameter(torch.zeros(hidden_dim)),
            "context": nn.Parameter(torch.zeros(hidden_dim)),
            "label": nn.Parameter(torch.zeros(hidden_dim)),
        }
        if self.multiscale:
            type_bias_dict["patch"] = nn.Parameter(torch.zeros(hidden_dim))
            
        self.type_bias = nn.ParameterDict(type_bias_dict)
        
        # --- Message Passing ---
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for _ in range(num_layers):
            conv_dict = {
                ("image", "has_context", "context"): GATConv((-1, -1), hidden_dim, heads=heads, concat=False, dropout=dropout, add_self_loops=False),
                ("context", "to", "image"): GATConv((-1, -1), hidden_dim, heads=heads, concat=False, dropout=dropout, add_self_loops=False),
                ("image", "similar_to", "image"): GATConv((-1, -1), hidden_dim, heads=heads, concat=False, dropout=dropout, add_self_loops=False),
                ("image", "same_patient", "image"): GATConv((-1, -1), hidden_dim, heads=heads, concat=False, dropout=dropout, add_self_loops=False),
            }
            
            if self.multiscale:
                conv_dict.update({
                    ("patch", "adjacent", "patch"): SAGEConv(-1, hidden_dim),
                    ("patch", "to", "image"): SAGEConv(-1, hidden_dim),
                    ("image", "to", "patch"): SAGEConv(-1, hidden_dim),
                })
                
            self.convs.append(HeteroConv(conv_dict, aggr="sum"))
            
            norm_dict = {
                "image": nn.LayerNorm(hidden_dim),
                "context": nn.LayerNorm(hidden_dim),
                "label": nn.LayerNorm(hidden_dim)
            }
            if self.multiscale:
                norm_dict["patch"] = nn.LayerNorm(hidden_dim)
                
            self.norms.append(nn.ModuleDict(norm_dict))
            
        self.out_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels, bias=False)
        )
        
        if initial_biases is not None:
            self.out_bias = nn.Parameter(torch.tensor(initial_biases, dtype=torch.float32))
        else:
            self.out_bias = nn.Parameter(torch.zeros(num_labels, dtype=torch.float32))
            
    def forward(self, data: HeteroData) -> torch.Tensor:
        x_dict = {}
        
        if self.multiscale:
            # Patch Features
            cnn_feats = data["patch"].x
            x_patch = self.patch_encoder(cnn_feats) + self.type_bias["patch"]
            x_dict["patch"] = x_patch
            
            # Image features start empty and get updated by patches
            num_images = data["image"].num_nodes
            x_img = torch.zeros((num_images, self.hidden_dim), device=cnn_feats.device) + self.type_bias["image"]
            x_dict["image"] = x_img
        else:
            # Image Features
            cnn_feats = data["image"].x
            x_img = self.image_encoder(cnn_feats) + self.type_bias["image"]
            x_dict["image"] = x_img
        
        # 2. Context & Label Features
        num_contexts = data["context"].num_nodes if "context" in data.node_types else 0
        if num_contexts > 0 and self.context_embed is not None:
            x_ctx = self.context_embed(data["context"].context_idx.long()) + self.type_bias["context"]
            x_dict["context"] = x_ctx
            
        x_lbl = self.label_embed(data["label"].label_idx.long()) + self.type_bias["label"]
        x_dict["label"] = x_lbl
        
        # 3. Message Passing
        for conv, norm in zip(self.convs, self.norms):
            out_dict = conv(x_dict, data.edge_index_dict)
            next_x = {}
            for node_type, x in x_dict.items():
                if node_type in out_dict:
                    h = F.relu(out_dict[node_type])
                    h = F.dropout(h, p=self.dropout, training=self.training)
                    next_x[node_type] = norm[node_type](x + h)
                else:
                    next_x[node_type] = x
            x_dict = next_x
            
        # 4. Readout
        logits = self.out_head(x_dict["image"]) + self.out_bias
        return logits
