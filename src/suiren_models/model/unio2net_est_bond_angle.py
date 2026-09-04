from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unio2net_est import GaussianSmearing, UniTransformerO2TwoUpdateGeneral


class AngularEncoding(nn.Module):
    """Simple sinusoidal angular encoding for triplet bond updates."""

    def __init__(self, num_freqs: int = 8) -> None:
        super().__init__()
        self.num_freqs = num_freqs
        freq = torch.arange(1, num_freqs + 1, dtype=torch.float32)
        self.register_buffer("freq", freq)

    def get_out_dim(self) -> int:
        return self.num_freqs * 2

    def forward(self, angle: torch.Tensor) -> torch.Tensor:
        angle = angle.unsqueeze(-1) * self.freq.view(1, -1)
        return torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)


class BondTripletAngleLayer(nn.Module):
    """
    MolPilot-style bond refinement with triplet and angle awareness.

    For each bond j->i, the layer aggregates messages from neighboring bonds k->j,
    using angle(i, j, k), bond lengths, and optional endpoint node states.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        num_r_gaussian: int,
        r_max: float,
        include_h_node: bool = True,
        dropout: float = 0.0,
        coord_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError("hidden_dim must be divisible by n_heads.")

        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.include_h_node = include_h_node
        self.coord_scale = coord_scale
        self.dropout = dropout

        self.distance_expansion = GaussianSmearing(
            start=0.0,
            stop=r_max,
            num_gaussians=num_r_gaussian,
            basis_width_scalar=2.0,
        )
        self.angle_expansion = AngularEncoding(num_freqs=8)

        kv_input_dim = hidden_dim + num_r_gaussian * 2 + self.angle_expansion.get_out_dim()
        q_input_dim = hidden_dim
        if include_h_node:
            kv_input_dim += hidden_dim * 2
            q_input_dim += hidden_dim

        self.lin_key = nn.Linear(kv_input_dim, hidden_dim, bias=False)
        self.lin_value = nn.Linear(kv_input_dim, hidden_dim, bias=False)
        self.lin_query = nn.Linear(q_input_dim, hidden_dim, bias=False)
        self.lin_edge0 = nn.Linear(num_r_gaussian, hidden_dim, bias=False)
        self.lin_edge1 = nn.Linear(num_r_gaussian, hidden_dim, bias=False)

        self.bond_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_from_bond = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.coord_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.bond_norm = nn.LayerNorm(hidden_dim)
        self.node_norm = nn.LayerNorm(hidden_dim)

    @staticmethod
    def triplets(
        bond_index: torch.Tensor,
        num_nodes: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        src, dst = bond_index
        device = bond_index.device

        incoming_edges = [[] for _ in range(num_nodes)]
        for edge_id in range(src.numel()):
            incoming_edges[dst[edge_id].item()].append(edge_id)

        idx_i = []
        idx_j = []
        idx_k = []
        idx_kj = []
        idx_ji = []

        for edge_ji in range(src.numel()):
            j = src[edge_ji].item()
            i = dst[edge_ji].item()
            for edge_kj in incoming_edges[j]:
                k = src[edge_kj].item()
                if k == i:
                    continue
                idx_i.append(i)
                idx_j.append(j)
                idx_k.append(k)
                idx_kj.append(edge_kj)
                idx_ji.append(edge_ji)

        if len(idx_i) == 0:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return dst, src, empty, empty, empty, empty, empty

        return (
            dst,
            src,
            torch.tensor(idx_i, dtype=torch.long, device=device),
            torch.tensor(idx_j, dtype=torch.long, device=device),
            torch.tensor(idx_k, dtype=torch.long, device=device),
            torch.tensor(idx_kj, dtype=torch.long, device=device),
            torch.tensor(idx_ji, dtype=torch.long, device=device),
        )

    @staticmethod
    def _segment_softmax(src: torch.Tensor, index: torch.Tensor, num_segments: int) -> torch.Tensor:
        out = torch.zeros_like(src)
        for seg_id in range(num_segments):
            mask = index == seg_id
            if torch.any(mask):
                out[mask] = torch.softmax(src[mask], dim=0)
        return out

    def forward(
        self,
        h: torch.Tensor,
        h_bond: torch.Tensor,
        pos: torch.Tensor,
        bond_index: torch.Tensor,
        mask_ligand: torch.Tensor,
        fix_x: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_nodes = h.size(0)
        num_bonds = h_bond.size(0)
        if num_bonds == 0 or bond_index.numel() == 0:
            return h_bond, torch.zeros_like(h), pos

        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = self.triplets(bond_index, num_nodes)
        src, dst = bond_index
        rel = pos[dst] - pos[src]
        dist = torch.norm(rel, p=2, dim=-1, keepdim=True).clamp_min(1e-8)
        r_feat = self.distance_expansion(dist)

        if idx_ji.numel() == 0:
            bond_delta = self.bond_out(h_bond)
            new_h_bond = self.bond_norm(h_bond + bond_delta)
            node_msg = self.node_from_bond(new_h_bond)
            node_delta = torch.zeros_like(h)
            node_delta.index_add_(0, src, node_msg)
            node_delta.index_add_(0, dst, node_msg)
            node_delta = self.node_norm(node_delta)
            if not fix_x:
                direction = rel / dist
                edge_scalar = torch.tanh(self.coord_gate(new_h_bond))
                edge_delta = self.coord_scale * edge_scalar * direction
                pos_delta = torch.zeros_like(pos)
                pos_delta.index_add_(0, src, -edge_delta)
                pos_delta.index_add_(0, dst, edge_delta)
                pos = pos + pos_delta * mask_ligand.to(pos.dtype).unsqueeze(-1)
            return new_h_bond, node_delta, pos

        pos_i = pos[idx_i]
        vec_ji = pos[idx_j] - pos_i
        vec_ki = pos[idx_k] - pos_i
        a = (vec_ji * vec_ki).sum(dim=-1)
        b = torch.cross(vec_ji, vec_ki, dim=-1).norm(dim=-1)
        angle = torch.atan2(b, a).clamp(min=0.0, max=math.pi)
        a_feat = self.angle_expansion(angle)

        hi = h[idx_i]
        hj = h[idx_j]
        hk = h[idx_k]
        h_bond_kj = h_bond[idx_kj]
        h_bond_ji = h_bond[idx_ji]
        r_feat_kj = r_feat[idx_kj]
        r_feat_ji = r_feat[idx_ji]

        kv_input = [h_bond_kj, r_feat_kj, r_feat_ji, a_feat]
        q_input = [h_bond_ji]
        if self.include_h_node:
            kv_input.extend([hk, hj])
            q_input.append(hi)
        kv_input = torch.cat(kv_input, dim=-1)
        q_input = torch.cat(q_input, dim=-1)

        key = self.lin_key(kv_input).view(-1, self.n_heads, self.head_dim)
        value = self.lin_value(kv_input).view(-1, self.n_heads, self.head_dim)
        query = self.lin_query(q_input).view(-1, self.n_heads, self.head_dim)
        edge_attn = torch.tanh(self.lin_edge0(r_feat_ji)).view(-1, self.n_heads, self.head_dim)

        alpha_logits = (query * key * edge_attn).sum(dim=-1) / math.sqrt(self.head_dim)
        alpha = self._segment_softmax(alpha_logits, idx_ji, num_bonds)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        msg_gate = torch.tanh(self.lin_edge1(r_feat_ji)).view(-1, self.n_heads, self.head_dim)
        msg = value * msg_gate
        msg = msg * alpha.unsqueeze(-1)

        bond_msg = torch.zeros(
            num_bonds, self.n_heads, self.head_dim, device=h.device, dtype=h.dtype
        )
        bond_msg.index_add_(0, idx_ji, msg)
        bond_msg = bond_msg.reshape(num_bonds, self.hidden_dim)

        bond_delta = self.bond_out(bond_msg)
        new_h_bond = self.bond_norm(h_bond + bond_delta)

        node_msg = self.node_from_bond(new_h_bond)
        node_delta = torch.zeros_like(h)
        node_delta.index_add_(0, src, node_msg)
        node_delta.index_add_(0, dst, node_msg)
        node_delta = self.node_norm(node_delta)

        if not fix_x:
            direction = rel / dist
            edge_scalar = torch.tanh(self.coord_gate(new_h_bond))
            edge_delta = self.coord_scale * edge_scalar * direction
            pos_delta = torch.zeros_like(pos)
            pos_delta.index_add_(0, src, -edge_delta)
            pos_delta.index_add_(0, dst, edge_delta)
            ligand_mask = mask_ligand.to(pos.dtype).unsqueeze(-1)
            pos = pos + pos_delta * ligand_mask

        return new_h_bond, node_delta, pos


class UniTransformerO2TwoUpdateGeneralBondAngle(UniTransformerO2TwoUpdateGeneral):
    """
    EST / Equiformer backbone + MolPilot-style triplet-angle bond refinement.

    Compatible interface:
        outputs = model(
            h, x, group_idx, bond_index, h_bond, mask_ligand, batch,
            node_time, bond_time, include_protein, return_all=False
        )
    """

    def __init__(
        self,
        num_blocks: int,
        num_layers: int,
        hidden_dim: int,
        n_heads: int = 1,
        knn: int = 32,
        num_bond_classes: int = 1,
        num_r_gaussian: int = 50,
        edge_feat_dim: int = 4,
        act_fn: str = "silu",
        norm: bool = True,
        cutoff_mode: str = "radius",
        use_global_ew: bool = True,
        adaptive_norm: bool = True,
        r_max: float = 10.0,
        x2h_out_fc: bool = True,
        sync_twoup: bool = False,
        h_node_in_bond_net: bool = True,
        name: str = "unio2_net_bond_est_angle",
        bond_net_type: str = "triplet_angle",
        dropout: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(
            num_blocks=num_blocks,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            knn=knn,
            num_r_gaussian=num_r_gaussian,
            edge_feat_dim=edge_feat_dim,
            act_fn=act_fn,
            norm=norm,
            cutoff_mode=cutoff_mode,
            r_max=r_max,
            x2h_out_fc=x2h_out_fc,
            sync_twoup=sync_twoup,
            name=name,
            **kwargs,
        )
        self.num_bond_classes = num_bond_classes
        self.use_global_ew = use_global_ew
        self.adaptive_norm = adaptive_norm
        self.h_node_in_bond_net = h_node_in_bond_net
        self.bond_net_type = bond_net_type
        self.dropout_prob = dropout

        self.bond_layers = nn.ModuleList(
            [
                BondTripletAngleLayer(
                    hidden_dim=self.hidden_dim,
                    n_heads=self.n_heads,
                    num_r_gaussian=self.num_r_gaussian,
                    r_max=self.r_max,
                    include_h_node=self.h_node_in_bond_net,
                    dropout=self.dropout_prob,
                    coord_scale=self.coord_scale,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.bond_fuse_norm = nn.ModuleList(
            [nn.LayerNorm(self.hidden_dim) for _ in range(self.num_layers)]
        )
        self.node_time_proj = nn.Linear(1, self.hidden_dim) if self.adaptive_norm else None
        self.bond_time_proj = nn.Linear(1, self.hidden_dim) if self.adaptive_norm else None

    def __repr__(self) -> str:
        return (
            f"UniTransformerO2TwoUpdateGeneralBondAngle(name={self.name}, num_blocks={self.num_blocks}, "
            f"num_layers={self.num_layers}, hidden_dim={self.hidden_dim}, n_heads={self.n_heads}, "
            f"cutoff_mode={self.cutoff_mode}, bond_net_type={self.bond_net_type}, "
            f"num_bond_classes={self.num_bond_classes})"
        )

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        group_idx,
        bond_index: torch.Tensor,
        h_bond: torch.Tensor,
        mask_ligand: torch.Tensor,
        batch: torch.Tensor,
        node_time: torch.Tensor | None,
        bond_time: torch.Tensor | None,
        include_protein: bool,
        return_all: bool = False,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        del group_idx

        current_h = h
        current_x = x
        current_h_bond = h_bond

        if self.adaptive_norm and node_time is not None:
            current_h = current_h + self.node_time_proj(node_time)
        if self.adaptive_norm and bond_time is not None and current_h_bond.numel() > 0:
            current_h_bond = current_h_bond + self.bond_time_proj(bond_time)

        all_x = [current_x]
        all_h = [current_h]
        all_h_bond = [current_h_bond]

        mask_ligand = mask_ligand.to(device=current_x.device)
        batch = batch.to(device=current_x.device)
        dummy_atomic_numbers = torch.zeros(
            current_x.size(0), dtype=torch.long, device=current_x.device
        )

        for _ in range(self.num_blocks):
            edge_index = None
            if include_protein:
                edge_index = self._connect_edge(current_x, batch)

            for layer_idx, bond_layer in enumerate(self.bond_layers):
                if include_protein:
                    x_embed = self._init_so3_embedding(current_h)
                    edge_scalar = self._build_edge_scalar(current_x, edge_index, mask_ligand)
                    x_embed, current_x = self.base_block[layer_idx](
                        x_embed=x_embed,
                        pos=current_x,
                        edge_index=edge_index,
                        edge_scalar=edge_scalar,
                        batch=batch,
                        mask_ligand=mask_ligand,
                        dummy_atomic_numbers=dummy_atomic_numbers,
                        fix_x=False,
                    )
                    current_h = self._decode_hidden(x_embed, current_h)

                current_h_bond, bond_node_msg, current_x = bond_layer(
                    h=current_h,
                    h_bond=current_h_bond,
                    pos=current_x,
                    bond_index=bond_index,
                    mask_ligand=mask_ligand,
                    fix_x=False,
                )
                current_h = self.bond_fuse_norm[layer_idx](current_h + bond_node_msg)

                if return_all:
                    all_x.append(current_x)
                    all_h.append(current_h)
                    all_h_bond.append(current_h_bond)

        outputs: dict[str, torch.Tensor | list[torch.Tensor]] = {
            "x": current_x,
            "h": current_h,
            "h_bond": current_h_bond,
        }
        if return_all:
            outputs["all_x"] = all_x
            outputs["all_h"] = all_h
            outputs["all_h_bond"] = all_h_bond
        return outputs
