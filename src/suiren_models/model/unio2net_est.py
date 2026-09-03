from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import knn_graph, radius_graph

from .edge_rot_mat import init_edge_rot_mat
from .input_block import EdgeDegreeEmbedding
from .so3 import CoefficientMappingModule, SO3_Embedding, SO3_Grid, SO3_Rotation
from .transformer_block import SO2EquivariantGraphAttention, ToS2Grid_block, TransBlockV2


class GaussianSmearing(nn.Module):
    """Gaussian radial basis expansion used by the original EST backbone."""

    def __init__(
        self,
        start: float = 0.0,
        stop: float = 10.0,
        num_gaussians: int = 50,
        basis_width_scalar: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_output = num_gaussians
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (basis_width_scalar * (offset[1] - offset[0])).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class ESTCoordinateUpdateLayer(nn.Module):
    """
    One EST update layer that:
    1. injects geometry-aware edge-degree features,
    2. updates hidden SO(3) features with TransBlockV2,
    3. predicts equivariant coordinate deltas from the updated representation.
    """

    def __init__(
        self,
        sphere_channels: int,
        attn_hidden_channels: int,
        num_heads: int,
        attn_alpha_channels: int,
        attn_value_channels: int,
        ffn_hidden_channels: int,
        lmax_list: list[int],
        mmax_list: list[int],
        so3_rotation,
        mapping_reduced,
        so3_grid,
        edge_channels_list: list[int],
        norm_type: str,
        attn_activation: str,
        ffn_activation: str,
        use_gate_act: bool,
        use_grid_mlp: bool,
        use_sep_s2_act: bool,
        alpha_drop: float,
        drop_path_rate: float,
        proj_drop: float,
        num_experts_steerable: int,
        num_experts_spherical: int,
        tos2grid,
        avg_degree: float,
        coord_scale: float,
    ) -> None:
        super().__init__()
        self.coord_scale = coord_scale
        self.edge_degree_embedding = EdgeDegreeEmbedding(
            sphere_channels=sphere_channels,
            lmax_list=lmax_list,
            mmax_list=mmax_list,
            SO3_rotation=so3_rotation,
            mappingReduced=mapping_reduced,
            max_num_elements=1,
            edge_channels_list=edge_channels_list,
            use_atom_edge_embedding=False,
            rescale_factor=avg_degree,
        )
        self.block = TransBlockV2(
            sphere_channels=sphere_channels,
            attn_hidden_channels=attn_hidden_channels,
            num_heads=num_heads,
            attn_alpha_channels=attn_alpha_channels,
            attn_value_channels=attn_value_channels,
            ffn_hidden_channels=ffn_hidden_channels,
            output_channels=sphere_channels,
            lmax_list=lmax_list,
            mmax_list=mmax_list,
            SO3_rotation=so3_rotation,
            mappingReduced=mapping_reduced,
            SO3_grid=so3_grid,
            max_num_elements=1,
            edge_channels_list=edge_channels_list,
            use_atom_edge_embedding=False,
            use_m_share_rad=False,
            attn_activation=attn_activation,
            use_s2_act_attn=False,
            use_attn_renorm=True,
            ffn_activation=ffn_activation,
            use_gate_act=use_gate_act,
            use_grid_mlp=use_grid_mlp,
            use_sep_s2_act=use_sep_s2_act,
            norm_type=norm_type,
            alpha_drop=alpha_drop,
            drop_path_rate=drop_path_rate,
            proj_drop=proj_drop,
            num_experts_steerable=num_experts_steerable,
            num_experts_spherical=num_experts_spherical,
            tos2grid=tos2grid,
        )
        self.coord_head = SO2EquivariantGraphAttention(
            sphere_channels=sphere_channels,
            hidden_channels=attn_hidden_channels,
            num_heads=num_heads,
            attn_alpha_channels=attn_alpha_channels,
            attn_value_channels=attn_value_channels,
            output_channels=1,
            lmax_list=lmax_list,
            mmax_list=mmax_list,
            SO3_rotation=so3_rotation,
            mappingReduced=mapping_reduced,
            SO3_grid=so3_grid,
            max_num_elements=1,
            edge_channels_list=edge_channels_list,
            use_atom_edge_embedding=False,
            use_m_share_rad=False,
            activation=attn_activation,
            use_s2_act_attn=False,
            use_attn_renorm=True,
            use_gate_act=use_gate_act,
            use_sep_s2_act=use_sep_s2_act,
            alpha_drop=0.0,
        )

    def forward(
        self,
        x_embed: SO3_Embedding,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_scalar: torch.Tensor,
        batch: torch.Tensor,
        mask_ligand: torch.Tensor,
        dummy_atomic_numbers: torch.Tensor,
        fix_x: bool = False,
    ) -> tuple[SO3_Embedding, torch.Tensor]:
        edge_vec = pos[edge_index[1]] - pos[edge_index[0]]
        edge_rot_mat = init_edge_rot_mat(edge_vec)
        for rotation in self.block.ga.SO3_rotation[0]:
            rotation.set_wigner(edge_rot_mat)

        block_input = x_embed.clone()
        edge_degree = self.edge_degree_embedding(
            dummy_atomic_numbers,
            edge_scalar,
            edge_index,
            num_nodes=pos.size(0),
            node_offset=0,
            ssp=False,
        )
        block_input.embedding = block_input.embedding + edge_degree.embedding

        x_embed = self.block(
            block_input,
            dummy_atomic_numbers,
            edge_scalar,
            edge_index,
            batch=batch,
            node_offset=0,
            ssp=False,
        )

        coord_update = self.coord_head(
            x_embed,
            dummy_atomic_numbers,
            edge_scalar,
            edge_index,
            node_offset=0,
            ssp=False,
        )
        coord_update = coord_update.embedding.narrow(1, 1, 3).reshape(-1, 3)

        if not fix_x:
            ligand_mask = mask_ligand.to(pos.dtype).unsqueeze(-1)
            pos = pos + self.coord_scale * coord_update * ligand_mask

        return x_embed, pos


class UniTransformerO2TwoUpdateGeneral(nn.Module):
    """
    Drop-in replacement for molcraft/TargetDiff-style ``unio2net``.

    Interface compatibility:
        outputs = model(h_all, pos_all, mask_ligand, batch_all, return_all=False, fix_x=False)
        final_pos, final_h = outputs["x"], outputs["h"]

    Design notes:
    - Hidden updates use the EST / EquiformerV2-style `TransBlockV2`.
    - Coordinate updates use an equivariant vector head adapted from the force head.
    - Protein nodes participate in message passing but only ligand coordinates are updated.
    """

    def __init__(
        self,
        num_blocks: int,
        num_layers: int,
        hidden_dim: int,
        n_heads: int = 4,
        knn: int = 32,
        num_r_gaussian: int = 50,
        edge_feat_dim: int = 4,
        num_node_types: int = 8,  # kept for config compatibility
        act_fn: str = "silu",  # kept for config compatibility
        norm: bool = True,
        cutoff_mode: str = "radius",
        ew_net_type: str = "r",  # kept for config compatibility
        num_init_x2h: int = 1,  # kept for config compatibility
        num_init_h2x: int = 0,  # kept for config compatibility
        num_x2h: int = 1,  # kept for config compatibility
        num_h2x: int = 1,  # kept for config compatibility
        r_max: float = 10.0,
        x2h_out_fc: bool = True,  # kept for config compatibility
        sync_twoup: bool = False,  # kept for config compatibility
        name: str = "unio2net",
        sphere_channels: int | None = None,
        attn_hidden_channels: int | None = None,
        attn_alpha_channels: int = 32,
        attn_value_channels: int = 16,
        ffn_hidden_channels: int | None = None,
        lmax_list: list[int] | None = None,
        mmax_list: list[int] | None = None,
        grid_resolution: int | None = None,
        EST_grid_resolution: int | None = None,
        EST_grid_optim_step: int = 0,
        edge_channels: int = 128,
        norm_type: str | None = None,
        attn_activation: str = "silu",
        ffn_activation: str = "silu",
        use_gate_act: bool = False,
        use_grid_mlp: bool = True,
        use_sep_s2_act: bool = True,
        alpha_drop: float = 0.0,
        drop_path_rate: float = 0.0,
        proj_drop: float = 0.0,
        num_experts_steerable: int = 4,
        num_experts_spherical: int = 4,
        avg_degree: float = 16.0,
        coord_scale: float = 1.0,
        max_num_neighbors: int = 128,
        **kwargs,
    ) -> None:
        super().__init__()
        del num_node_types, act_fn, ew_net_type, num_init_x2h, num_init_h2x
        del num_x2h, num_h2x, x2h_out_fc, sync_twoup, kwargs

        if lmax_list is None:
            lmax_list = [4]
        if mmax_list is None:
            mmax_list = [2]
        if max(lmax_list) < 1:
            raise ValueError("Coordinate prediction requires lmax >= 1.")

        self.name = name
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.knn = knn
        self.cutoff_mode = cutoff_mode
        self.r_max = r_max
        self.max_num_neighbors = max_num_neighbors
        self.num_r_gaussian = num_r_gaussian
        self.edge_feat_dim = edge_feat_dim
        self.sphere_channels = sphere_channels or hidden_dim
        self.attn_hidden_channels = attn_hidden_channels or hidden_dim
        self.ffn_hidden_channels = ffn_hidden_channels or (hidden_dim * 4)
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.num_resolutions = len(self.lmax_list)
        self.scalar_slots = self.num_resolutions * self.sphere_channels
        self.coord_scale = coord_scale
        self.norm_type = norm_type or ("layer_norm_sh" if norm else "layer_norm")

        self.distance_expansion = GaussianSmearing(
            start=0.0,
            stop=r_max,
            num_gaussians=num_r_gaussian,
            basis_width_scalar=2.0,
        )

        if self.edge_feat_dim > 0:
            if self.edge_feat_dim == 4:
                self.edge_type_proj = nn.Identity()
            else:
                self.edge_type_proj = nn.Linear(4, self.edge_feat_dim, bias=False)
            scalar_edge_input_dim = self.num_r_gaussian * self.edge_feat_dim
        else:
            self.edge_type_proj = None
            scalar_edge_input_dim = self.num_r_gaussian
        self.edge_channels_list = [scalar_edge_input_dim] + [edge_channels] * 2

        self.input_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.scalar_slots),
        )
        self.output_proj = nn.Sequential(
            nn.Linear(self.scalar_slots, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

        self.so3_rotation_base = nn.ModuleList(
            [SO3_Rotation(lmax) for lmax in self.lmax_list]
        )
        self.so3_rotation = [self.so3_rotation_base, self.so3_rotation_base]
        self.mapping_reduced = CoefficientMappingModule(self.lmax_list, self.mmax_list)

        self.so3_grid = nn.ModuleList()
        max_l = max(self.lmax_list)
        for lval in range(max_l + 1):
            grids_for_l = nn.ModuleList()
            for mval in range(max_l + 1):
                grids_for_l.append(
                    SO3_Grid(
                        lval,
                        mval,
                        normalization="component",
                        resolution=grid_resolution,
                    )
                )
            self.so3_grid.append(grids_for_l)

        if EST_grid_resolution is not None:
            self.tos2grid = ToS2Grid_block(
                max_l,
                EST_grid_resolution,
                optim_step=EST_grid_optim_step,
            )
        else:
            default_est_resolution = (2 * max_l + 1) * (2 * max_l + 2)
            self.tos2grid = ToS2Grid_block(
                max_l,
                default_est_resolution,
                optim_step=EST_grid_optim_step,
            )

        self.base_block = nn.ModuleList(
            [
                ESTCoordinateUpdateLayer(
                    sphere_channels=self.sphere_channels,
                    attn_hidden_channels=self.attn_hidden_channels,
                    num_heads=self.n_heads,
                    attn_alpha_channels=attn_alpha_channels,
                    attn_value_channels=attn_value_channels,
                    ffn_hidden_channels=self.ffn_hidden_channels,
                    lmax_list=self.lmax_list,
                    mmax_list=self.mmax_list,
                    so3_rotation=self.so3_rotation,
                    mapping_reduced=self.mapping_reduced,
                    so3_grid=self.so3_grid,
                    edge_channels_list=self.edge_channels_list,
                    norm_type=self.norm_type,
                    attn_activation=attn_activation,
                    ffn_activation=ffn_activation,
                    use_gate_act=use_gate_act,
                    use_grid_mlp=use_grid_mlp,
                    use_sep_s2_act=use_sep_s2_act,
                    alpha_drop=alpha_drop,
                    drop_path_rate=drop_path_rate,
                    proj_drop=proj_drop,
                    num_experts_steerable=num_experts_steerable,
                    num_experts_spherical=num_experts_spherical,
                    tos2grid=self.tos2grid,
                    avg_degree=avg_degree,
                    coord_scale=self.coord_scale,
                )
                for _ in range(self.num_layers)
            ]
        )

    def __repr__(self) -> str:
        return (
            f"UniTransformerO2TwoUpdateGeneral(name={self.name}, num_blocks={self.num_blocks}, "
            f"num_layers={self.num_layers}, hidden_dim={self.hidden_dim}, n_heads={self.n_heads}, "
            f"cutoff_mode={self.cutoff_mode}, r_max={self.r_max}, lmax_list={self.lmax_list}, "
            f"mmax_list={self.mmax_list})"
        )

    def _connect_edge(
        self,
        x: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        if self.cutoff_mode == "radius":
            edge_index = radius_graph(
                x,
                r=self.r_max,
                batch=batch,
                loop=False,
                max_num_neighbors=self.max_num_neighbors,
                flow="source_to_target",
            )
        elif self.cutoff_mode == "knn":
            edge_index = knn_graph(
                x,
                k=self.knn,
                batch=batch,
                loop=False,
                flow="source_to_target",
            )
        elif self.cutoff_mode == "hybrid":
            edge_radius = radius_graph(
                x,
                r=self.r_max,
                batch=batch,
                loop=False,
                max_num_neighbors=self.max_num_neighbors,
                flow="source_to_target",
            )
            edge_knn = knn_graph(
                x,
                k=self.knn,
                batch=batch,
                loop=False,
                flow="source_to_target",
            )
            edge_index = torch.cat([edge_radius, edge_knn], dim=1)
            edge_index = torch.unique(edge_index, dim=1)
        else:
            raise ValueError(f"Unsupported cutoff mode: {self.cutoff_mode}")
        return edge_index

    @staticmethod
    def _build_edge_type(
        edge_index: torch.Tensor,
        mask_ligand: torch.Tensor,
    ) -> torch.Tensor:
        src, dst = edge_index
        src_ligand = mask_ligand[src].bool()
        dst_ligand = mask_ligand[dst].bool()

        edge_type = torch.zeros(edge_index.size(1), dtype=torch.long, device=edge_index.device)
        edge_type[src_ligand & dst_ligand] = 0
        edge_type[src_ligand & ~dst_ligand] = 1
        edge_type[~src_ligand & dst_ligand] = 2
        edge_type[~src_ligand & ~dst_ligand] = 3
        return F.one_hot(edge_type, num_classes=4).to(torch.float32)

    @staticmethod
    def _outer_product(edge_feat: torch.Tensor, radial_feat: torch.Tensor) -> torch.Tensor:
        prod = edge_feat.unsqueeze(-1) * radial_feat.unsqueeze(-2)
        return prod.reshape(edge_feat.size(0), -1)

    def _build_edge_scalar(
        self,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        mask_ligand: torch.Tensor,
    ) -> torch.Tensor:
        src, dst = edge_index
        rel = pos[dst] - pos[src]
        dist = torch.norm(rel, p=2, dim=-1, keepdim=True)
        radial_feat = self.distance_expansion(dist)

        if self.edge_type_proj is None:
            return radial_feat

        edge_type = self._build_edge_type(edge_index, mask_ligand)
        edge_feat = self.edge_type_proj(edge_type)
        return self._outer_product(edge_feat, radial_feat)

    def _init_so3_embedding(self, h: torch.Tensor) -> SO3_Embedding:
        node_embed = SO3_Embedding(
            length=h.size(0),
            lmax_list=self.lmax_list,
            num_channels=self.sphere_channels,
            device=h.device,
            dtype=h.dtype,
        )
        projected = self.input_proj(h)

        coeff_offset = 0
        scalar_offset = 0
        for lmax in self.lmax_list:
            node_embed.embedding[:, coeff_offset, :] = projected[
                :, scalar_offset : scalar_offset + self.sphere_channels
            ]
            coeff_offset += (lmax + 1) ** 2
            scalar_offset += self.sphere_channels
        return node_embed

    def _decode_hidden(self, x_embed: SO3_Embedding, residual_h: torch.Tensor) -> torch.Tensor:
        scalar_features = []
        coeff_offset = 0
        for lmax in self.lmax_list:
            scalar_features.append(x_embed.embedding[:, coeff_offset, :])
            coeff_offset += (lmax + 1) ** 2
        scalar_features = torch.cat(scalar_features, dim=-1)
        hidden = self.output_proj(scalar_features)
        return self.output_norm(hidden + residual_h)

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        mask_ligand: torch.Tensor,
        batch: torch.Tensor,
        return_all: bool = False,
        fix_x: bool = False,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        mask_ligand = mask_ligand.to(device=x.device)
        batch = batch.to(device=x.device)
        dummy_atomic_numbers = torch.zeros(
            x.size(0), dtype=torch.long, device=x.device
        )

        all_x = [x]
        all_h = [h]

        x_embed = self._init_so3_embedding(h)
        current_pos = x
        current_h = h

        for _ in range(self.num_blocks):
            edge_index = self._connect_edge(current_pos, batch)

            for layer in self.base_block:
                edge_scalar = self._build_edge_scalar(current_pos, edge_index, mask_ligand)
                x_embed, current_pos = layer(
                    x_embed=x_embed,
                    pos=current_pos,
                    edge_index=edge_index,
                    edge_scalar=edge_scalar,
                    batch=batch,
                    mask_ligand=mask_ligand,
                    dummy_atomic_numbers=dummy_atomic_numbers,
                    fix_x=fix_x,
                )

            current_h = self._decode_hidden(x_embed, current_h)
            if return_all:
                all_x.append(current_pos)
                all_h.append(current_h)

        outputs: dict[str, torch.Tensor | list[torch.Tensor]] = {
            "x": current_pos,
            "h": current_h,
        }
        if return_all:
            outputs["all_x"] = all_x
            outputs["all_h"] = all_h
        return outputs


class TimeConditioning(nn.Module):
    """Lightweight time conditioning for node/bond hidden states."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t: torch.Tensor | None, target_dim: int) -> torch.Tensor | None:
        if t is None:
            return None
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        if t.size(-1) != 1:
            t = t[..., :1]
        emb = self.net(t)
        if emb.size(-1) != target_dim:
            raise ValueError(f"Time embedding dim {emb.size(-1)} != target dim {target_dim}")
        return emb


class BondRefineLayer(nn.Module):
    """
    Bond branch used together with the EST node backbone.

    It updates bond hidden states from endpoint node states and bond geometry, then
    feeds bond messages back to nodes and predicts a bond-induced coordinate delta.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_r_gaussian: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.distance_expansion = GaussianSmearing(
            start=0.0,
            stop=10.0,
            num_gaussians=num_r_gaussian,
            basis_width_scalar=2.0,
        )
        bond_input_dim = hidden_dim * 3 + num_r_gaussian
        self.bond_update = nn.Sequential(
            nn.Linear(bond_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_message = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.coord_gate = nn.Sequential(
            nn.Linear(hidden_dim + num_r_gaussian, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.norm_bond = nn.LayerNorm(hidden_dim)
        self.norm_node = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h_node: torch.Tensor,
        pos: torch.Tensor,
        h_bond: torch.Tensor,
        bond_index: torch.Tensor,
        mask_ligand: torch.Tensor,
        fix_x: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_nodes = h_node.size(0)
        if bond_index.numel() == 0 or h_bond.numel() == 0:
            zero_node = torch.zeros_like(h_node)
            zero_pos = torch.zeros_like(pos)
            return zero_node, h_bond, zero_pos

        src, dst = bond_index
        rel = pos[dst] - pos[src]
        dist = torch.norm(rel, p=2, dim=-1, keepdim=True)
        dist_feat = self.distance_expansion(dist)

        bond_input = torch.cat([h_node[src], h_node[dst], h_bond, dist_feat], dim=-1)
        bond_delta = self.bond_update(bond_input)
        h_bond = self.norm_bond(h_bond + bond_delta)

        node_msg = self.node_message(h_bond)
        node_ctx = torch.zeros_like(h_node)
        node_ctx.index_add_(0, src, node_msg)
        node_ctx.index_add_(0, dst, node_msg)

        degree = torch.zeros(num_nodes, 1, device=h_node.device, dtype=h_node.dtype)
        one = torch.ones(src.size(0), 1, device=h_node.device, dtype=h_node.dtype)
        degree.index_add_(0, src, one)
        degree.index_add_(0, dst, one)
        node_ctx = node_ctx / degree.clamp_min(1.0)
        node_ctx = self.norm_node(node_ctx)

        bond_gate = torch.tanh(self.coord_gate(torch.cat([h_bond, dist_feat], dim=-1)))
        bond_delta_pos = bond_gate * rel
        delta_pos = torch.zeros_like(pos)
        delta_pos.index_add_(0, dst, bond_delta_pos)
        delta_pos.index_add_(0, src, -bond_delta_pos)
        if fix_x:
            delta_pos = torch.zeros_like(delta_pos)
        else:
            delta_pos = delta_pos * mask_ligand.to(delta_pos.dtype).unsqueeze(-1)

        return node_ctx, h_bond, delta_pos


class UniTransformerO2TwoUpdateGeneralBond(UniTransformerO2TwoUpdateGeneral):
    """
    EST / Equiformer based backbone with an additional bond branch.

    Compatible target interface:
        outputs = model(
            h, x, group_idx, bond_index, h_bond, mask_ligand, batch,
            node_time, bond_time, include_protein, return_all=False
        )
        x, h, h_bond = outputs["x"], outputs["h"], outputs["h_bond"]
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
        name: str = "unio2_net_bond",
        bond_net_type: str = "mlp",
        dropout: float = 0.1,
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
        self.dropout = dropout

        if self.use_global_ew:
            self.edge_pred_layer = nn.Sequential(
                nn.Linear(num_r_gaussian, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.edge_pred_layer = None

        self.bond_layers = nn.ModuleList(
            [
                BondRefineLayer(
                    hidden_dim=hidden_dim,
                    num_r_gaussian=num_r_gaussian,
                    dropout=dropout,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.node_bond_fuse = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(self.num_layers)
            ]
        )
        self.node_time = TimeConditioning(hidden_dim) if adaptive_norm else None
        self.bond_time = TimeConditioning(hidden_dim) if adaptive_norm else None

    def _inject_hidden_into_so3(
        self,
        x_embed: SO3_Embedding,
        h: torch.Tensor,
    ) -> SO3_Embedding:
        projected = self.input_proj(h)
        coeff_offset = 0
        scalar_offset = 0
        for lmax in self.lmax_list:
            x_embed.embedding[:, coeff_offset, :] = projected[
                :, scalar_offset : scalar_offset + self.sphere_channels
            ]
            coeff_offset += (lmax + 1) ** 2
            scalar_offset += self.sphere_channels
        return x_embed

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        group_idx: torch.Tensor | None,
        bond_index: torch.Tensor,
        h_bond: torch.Tensor,
        mask_ligand: torch.Tensor,
        batch: torch.Tensor,
        node_time: torch.Tensor | None,
        bond_time: torch.Tensor | None,
        include_protein: bool,
        return_all: bool = False,
        fix_x: bool = False,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        del group_idx

        mask_ligand = mask_ligand.to(device=x.device)
        batch = batch.to(device=x.device)
        dummy_atomic_numbers = torch.zeros(
            x.size(0), dtype=torch.long, device=x.device
        )

        all_x = [x]
        all_h = [h]
        all_h_bond = [h_bond]

        current_pos = x
        current_h = h
        current_h_bond = h_bond
        x_embed = self._init_so3_embedding(current_h)

        node_time_emb = self.node_time(node_time, self.hidden_dim) if self.node_time else None
        bond_time_emb = self.bond_time(bond_time, self.hidden_dim) if self.bond_time else None

        for _ in range(self.num_blocks):
            if include_protein:
                edge_index = self._connect_edge(current_pos, batch)
                src, dst = edge_index
                if self.edge_pred_layer is not None:
                    dist = torch.norm(current_pos[dst] - current_pos[src], p=2, dim=-1, keepdim=True)
                    dist_feat = self.distance_expansion(dist)
                    e_w = torch.sigmoid(self.edge_pred_layer(dist_feat))
                else:
                    e_w = None
            else:
                edge_index = None
                e_w = None

            for layer_idx, layer in enumerate(self.base_block):
                if node_time_emb is not None:
                    current_h = current_h + node_time_emb
                if bond_time_emb is not None and current_h_bond.numel() > 0:
                    current_h_bond = current_h_bond + bond_time_emb

                if include_protein and edge_index is not None:
                    edge_scalar = self._build_edge_scalar(current_pos, edge_index, mask_ligand)
                    if e_w is not None:
                        edge_scalar = edge_scalar * e_w
                    x_embed, current_pos = layer(
                        x_embed=x_embed,
                        pos=current_pos,
                        edge_index=edge_index,
                        edge_scalar=edge_scalar,
                        batch=batch,
                        mask_ligand=mask_ligand,
                        dummy_atomic_numbers=dummy_atomic_numbers,
                        fix_x=fix_x,
                    )
                    current_h = self._decode_hidden(x_embed, current_h)
                else:
                    x_embed = self._inject_hidden_into_so3(x_embed, current_h)

                bond_ctx, current_h_bond, bond_delta_pos = self.bond_layers[layer_idx](
                    h_node=current_h,
                    pos=current_pos,
                    h_bond=current_h_bond,
                    bond_index=bond_index,
                    mask_ligand=mask_ligand,
                    fix_x=fix_x,
                )
                fused = self.node_bond_fuse[layer_idx](torch.cat([current_h, bond_ctx], dim=-1))
                current_h = current_h + fused
                current_pos = current_pos + bond_delta_pos
                x_embed = self._inject_hidden_into_so3(x_embed, current_h)

                if return_all:
                    all_x.append(current_pos)
                    all_h.append(current_h)
                    all_h_bond.append(current_h_bond)

        outputs: dict[str, torch.Tensor | list[torch.Tensor]] = {
            "x": current_pos,
            "h": current_h,
            "h_bond": current_h_bond,
        }
        if return_all:
            outputs["all_x"] = all_x
            outputs["all_h"] = all_h
            outputs["all_h_bond"] = all_h_bond
        return outputs
