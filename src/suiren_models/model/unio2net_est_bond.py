from __future__ import annotations

import torch
import torch.nn as nn

from .unio2net_est import GaussianSmearing, UniTransformerO2TwoUpdateGeneral


class BondRefinementLayer(nn.Module):
    """
    Bond-aware refinement branch.

    The layer updates:
    - `h_bond`: latent bond features used by the outer model to decode bond classes.
    - node hidden states through aggregated bond messages.
    - coordinates through invariant bond scalars times equivariant bond directions.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_r_gaussian: int,
        r_max: float,
        mlp_ratio: int = 2,
        dropout: float = 0.0,
        include_h_node: bool = True,
        coord_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.include_h_node = include_h_node
        self.coord_scale = coord_scale
        self.distance_expansion = GaussianSmearing(
            start=0.0,
            stop=r_max,
            num_gaussians=num_r_gaussian,
            basis_width_scalar=2.0,
        )

        bond_input_dim = hidden_dim + num_r_gaussian
        if include_h_node:
            bond_input_dim += 2 * hidden_dim

        self.bond_norm = nn.LayerNorm(hidden_dim)
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.bond_mlp = nn.Sequential(
            nn.Linear(bond_input_dim, hidden_dim * mlp_ratio),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * mlp_ratio, hidden_dim),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * mlp_ratio),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * mlp_ratio, hidden_dim),
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        h: torch.Tensor,
        h_bond: torch.Tensor,
        pos: torch.Tensor,
        bond_index: torch.Tensor,
        mask_ligand: torch.Tensor,
        fix_x: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if h_bond.numel() == 0 or bond_index.numel() == 0:
            return h_bond, torch.zeros_like(h), pos

        src, dst = bond_index
        rel = pos[dst] - pos[src]
        dist = torch.norm(rel, p=2, dim=-1, keepdim=True).clamp_min(1e-8)
        dist_feat = self.distance_expansion(dist)

        bond_inputs = [h_bond, dist_feat]
        if self.include_h_node:
            bond_inputs.extend([h[src], h[dst]])
        bond_inputs = torch.cat(bond_inputs, dim=-1)

        bond_delta = self.bond_mlp(bond_inputs)
        new_h_bond = self.bond_norm(h_bond + self.dropout(bond_delta))

        node_msg = self.node_mlp(new_h_bond)
        node_delta = torch.zeros_like(h)
        node_delta.index_add_(0, src, node_msg)
        node_delta.index_add_(0, dst, node_msg)
        node_delta = self.node_norm(node_delta)

        if not fix_x:
            direction = rel / dist
            edge_scalar = torch.tanh(self.coord_mlp(new_h_bond))
            edge_delta = self.coord_scale * edge_scalar * direction
            pos_delta = torch.zeros_like(pos)
            pos_delta.index_add_(0, src, -edge_delta)
            pos_delta.index_add_(0, dst, edge_delta)
            ligand_mask = mask_ligand.to(pos.dtype).unsqueeze(-1)
            pos = pos + pos_delta * ligand_mask

        return new_h_bond, node_delta, pos


class UniTransformerO2TwoUpdateGeneralBond(UniTransformerO2TwoUpdateGeneral):
    """
    Bond-aware EST / Equiformer backbone.

    Compatible interface:
        outputs = model(
            h, x, group_idx, bond_index, h_bond, mask_ligand, batch,
            node_time, bond_time, include_protein, return_all=False
        )

    Output keys:
        - `x`: updated coordinates
        - `h`: updated node features
        - `h_bond`: updated bond features
        - optional `all_x`, `all_h`, `all_h_bond`
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
        h_node_in_bond_net: bool = False,
        name: str = "unio2_net_bond_est",
        bond_net_type: str = "mlp",
        dropout: float = 0.0,
        mlp_ratio: int = 2,
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
        self.mlp_ratio = mlp_ratio

        self.bond_layers = nn.ModuleList(
            [
                BondRefinementLayer(
                    hidden_dim=self.hidden_dim,
                    num_r_gaussian=self.num_r_gaussian,
                    r_max=self.r_max,
                    mlp_ratio=self.mlp_ratio,
                    dropout=self.dropout_prob,
                    include_h_node=self.h_node_in_bond_net,
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
            f"UniTransformerO2TwoUpdateGeneralBond(name={self.name}, num_blocks={self.num_blocks}, "
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

                if self.bond_net_type != "none":
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
