"""
SpatialConsistencyClassifier

Takes a spatial region graph and outputs a single score:
  > 0.5 = corrupted (geometric contradiction detected)
  < 0.5 = consistent (geometry is valid)

Architecture:
  1. Node encoder: encodes region position and normal into latent vector
  2. SpatialRefFrameCalc: builds antisymmetric reference frame per edge
  3. Edge encoder: projects boundary vectors onto reference frame
  4. Message passing: aggregates edge messages into node embeddings
  5. Graph pooling: mean pool node embeddings into graph embedding
  6. Classifier: single sigmoid output

The model MUST use the antisymmetric reference frame to detect corruption.
A model that ignores it cannot detect that boundary vectors are inconsistent
with positions — it would have to memorize graph structure instead.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
import sys
sys.path.insert(0, '/workspaces/dygna')
from model.model import SpatialRefFrameCalc
from utils.utils import build_mlp_d


class SpatialConsistencyClassifier(nn.Module):
    def __init__(self, latent_size=64, mlp_layers=2):
        super().__init__()
        self.spatial_ref = SpatialRefFrameCalc()

        # Node encoder: position (3) + normal (3) = 6 input features
        self.node_encoder = build_mlp_d(
            6, latent_size, latent_size,
            num_layers=mlp_layers, lay_norm=True
        )

        # Edge encoder: projected boundary vec (3) = 3 input features
        # We project the boundary vector onto the local reference frame
        self.edge_encoder = build_mlp_d(
            3, latent_size, latent_size,
            num_layers=mlp_layers, lay_norm=True
        )

        # Message aggregation: sender + receiver + edge = 3 * latent_size
        self.message_encoder = build_mlp_d(
            3 * latent_size, latent_size, latent_size,
            num_layers=mlp_layers, lay_norm=True
        )

        # Graph classifier: latent -> scalar
        self.classifier = build_mlp_d(
            latent_size, latent_size, 1,
            num_layers=mlp_layers, lay_norm=False,
            use_sigmoid=True
        )

        self.eps = 1e-8

    def forward(self, graph):
        pos = graph.pos.float()
        normals = graph.normals.float()
        boundary_vecs = graph.boundary_vecs.float()
        edge_index = graph.edge_index.long()
        batch = graph.batch if hasattr(graph, 'batch') and graph.batch is not None \
                else torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device)

        senders, receivers = edge_index

        # 1. Node encoding
        node_feat = torch.cat([pos, normals], dim=1)  # (N, 6)
        node_latent = self.node_encoder(node_feat)     # (N, latent)

        # 2. Build antisymmetric reference frame per edge
        va, vb, vc = self.spatial_ref(
            edge_index,
            pos[senders], pos[receivers],
            normals[senders], normals[receivers],
            boundary_vecs[senders], boundary_vecs[receivers]
        )

        # 3. Project boundary vectors onto local reference frame
        # This is where the antisymmetry does work:
        # a corrupted boundary vector will project inconsistently
        # onto the frame built from positions and normals
        basis = torch.stack([va, vb, vc], dim=1)  # (E, 3, 3)
        bvec = boundary_vecs[senders]              # (E, 3)
        projected = torch.bmm(basis, bvec.unsqueeze(-1)).squeeze(-1)  # (E, 3)

        # 4. Edge encoding
        edge_latent = self.edge_encoder(projected)  # (E, latent)

        # 5. Message passing: one round
        msg_input = torch.cat([
            node_latent[senders],
            node_latent[receivers],
            edge_latent
        ], dim=1)  # (E, 3*latent)
        messages = self.message_encoder(msg_input)  # (E, latent)

        # Aggregate messages to nodes (sum)
        num_nodes = pos.shape[0]
        agg = torch.zeros(num_nodes, messages.shape[1],
                         device=messages.device)
        agg.index_add_(0, receivers, messages)

        # Residual: add to node latent
        node_out = node_latent + agg  # (N, latent)

        # 6. Graph-level pooling
        graph_embed = global_mean_pool(node_out, batch)  # (B, latent)

        # 7. Classification
        score = self.classifier(graph_embed)  # (B, 1)
        return score.squeeze(-1)              # (B,)
