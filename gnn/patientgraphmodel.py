import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

try:
    from torch_geometric.data import HeteroData
    from torch_geometric.nn import GATConv, HeteroConv
except ImportError as exc:
    raise SystemExit("Missing dependency: torch-geometric. Install it before using this model.") from exc


@dataclass
class DemographicSpec:
    name: str
    kind: str  # numeric or categorical
    mean: float = 0.0
    std: float = 1.0
    vocab: Optional[Dict[str, int]] = None


class PatientGraphModel(nn.Module):
    """Heterogeneous Graph Neural Network for Patient-Level Predictions."""
    def __init__(
        self,
        patch_dim: int,
        num_labels: int,
        demographic_specs: Sequence[DemographicSpec],
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        max_images_per_patient: int = 32,
        heads: int = 2,
    ) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.hidden_dim = hidden_dim
        self.demographic_specs = list(demographic_specs)
        self.dropout = dropout
        self.max_images_per_patient = max_images_per_patient

        # --- Node Encoders ---
        self.patch_encoder = nn.Sequential(
            nn.Linear(patch_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # --- Demographic Encoders ---
        self.demo_field_embed = nn.Embedding(len(self.demographic_specs), hidden_dim)
        self.demo_kind_embed = nn.Embedding(2, hidden_dim)
        self.demo_numeric = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.demo_cat_tables = nn.ModuleList()
        self.cat_field_to_table: Dict[int, int] = {}
        cat_count = 0
        for field_idx, spec in enumerate(self.demographic_specs):
            if spec.kind == "categorical":
                size = len(spec.vocab or {"__UNK__": 0})
                self.demo_cat_tables.append(nn.Embedding(size, hidden_dim))
                self.cat_field_to_table[field_idx] = cat_count
                cat_count += 1

        self.image_idx_embed = nn.Embedding(max_images_per_patient + 1, hidden_dim)
        self.eye_idx_embed = nn.Embedding(4, hidden_dim)
        self.patient_token = nn.Parameter(torch.zeros(1, hidden_dim))
        self.label_token = nn.Embedding(num_labels, hidden_dim)

        self.type_bias = nn.ParameterDict(
            {
                "patch": nn.Parameter(torch.zeros(hidden_dim)),
                "image": nn.Parameter(torch.zeros(hidden_dim)),
                "demographic": nn.Parameter(torch.zeros(hidden_dim)),
                "patient": nn.Parameter(torch.zeros(hidden_dim)),
                "label": nn.Parameter(torch.zeros(hidden_dim)),
            }
        )

        # --- Heterogeneous Convolutions ---
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    # Homogeneous edges CAN have self-loops
                    ("patch", "adjacent", "patch"): GATConv((-1, -1), hidden_dim, heads=heads, concat=False, dropout=dropout, add_self_loops=True),
                    ("label", "correlates", "label"): GATConv((-1, -1), hidden_dim, heads=heads, concat=False, dropout=dropout, add_self_loops=True),
                    
                    # Heterogeneous edges MUST NOT have self-loops
                    ("patch", "to", "image"): GATConv((-1, -1), hidden_dim, heads=heads, concat=False, dropout=dropout, add_self_loops=False),
                    ("image", "to", "patch"): GATConv((-1, -1), hidden_dim, heads=heads, concat=False, dropout=dropout, add_self_loops=False),
                    ("image", "to", "patient"): GATConv((-1, -1), hidden_dim, heads=heads, concat=False, dropout=dropout, add_self_loops=False),
                    ("patient", "to", "image"): GATConv((-1, -1), hidden_dim, heads=heads, concat=False, dropout=dropout, add_self_loops=False),
                    ("demographic", "to", "patient"): GATConv((-1, -1), hidden_dim, heads=heads, concat=False, dropout=dropout, add_self_loops=False),
                    ("patient", "to", "demographic"): GATConv((-1, -1), hidden_dim, heads=heads, concat=False, dropout=dropout, add_self_loops=False),
                    ("patient", "to", "label"): GATConv((-1, -1), hidden_dim, heads=heads, concat=False, dropout=dropout, add_self_loops=False),
                    ("label", "to", "patient"): GATConv((-1, -1), hidden_dim, heads=heads, concat=False, dropout=dropout, add_self_loops=False),
                },
                aggr="sum",
            )
            self.convs.append(conv)
            self.norms.append(
                nn.ModuleDict(
                    {
                        "patch": nn.LayerNorm(hidden_dim),
                        "image": nn.LayerNorm(hidden_dim),
                        "demographic": nn.LayerNorm(hidden_dim),
                        "patient": nn.LayerNorm(hidden_dim),
                        "label": nn.LayerNorm(hidden_dim),
                    }
                )
            )

        self.label_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def encode_demographics(self, data: HeteroData) -> torch.Tensor:
        field_idx = data["demographic"].field_idx.long()
        kind = data["demographic"].kind.long()
        num_value = data["demographic"].num_value.float()
        cat_value = data["demographic"].cat_value.long()
        missing = data["demographic"].missing.float()

        out = torch.zeros((field_idx.size(0), self.hidden_dim), device=field_idx.device)

        for i, spec in enumerate(self.demographic_specs):
            mask = field_idx == i
            if not mask.any():
                continue

            base = self.demo_field_embed(field_idx[mask]) + self.demo_kind_embed(kind[mask])
            if spec.kind == "numeric":
                emb = self.demo_numeric(torch.cat([num_value[mask], missing[mask]], dim=-1))
            else:
                emb = self.demo_cat_tables[self.cat_field_to_table[i]](cat_value[mask])
            out[mask] = base + emb

        return out + self.type_bias["demographic"]

    def encode_images(self, data: HeteroData) -> torch.Tensor:
        image_idx = data["image"].image_idx.long().clamp(min=0, max=self.max_images_per_patient)
        eye_idx = data["image"].eye_idx.long().clamp(min=0, max=3)
        return self.image_idx_embed(image_idx) + self.eye_idx_embed(eye_idx) + self.type_bias["image"]

    def forward(self, data: HeteroData) -> torch.Tensor:
        x_dict = {
            "patch": self.patch_encoder(data["patch"].x.float()) + self.type_bias["patch"],
            "image": self.encode_images(data),
            "demographic": self.encode_demographics(data),
            "patient": self.patient_token.expand(data["patient"].num_nodes, -1) + self.type_bias["patient"],
            "label": self.label_token(data["label"].label_idx.long()) + self.type_bias["label"],
        }

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

        logits = self.label_head(x_dict["label"]).squeeze(-1)
        num_graphs = data["patient"].num_nodes
        return logits.view(num_graphs, self.num_labels)