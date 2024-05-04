from kan import KAN

from typing import Literal, Optional
from torch import Tensor, nn

from nerfstudio.fields.nerfacto_field import NerfactoField
from nerfstudio.field_components.spatial_distortions import SpatialDistortion

class KANeRFactoField(NerfactoField):
    """Compound Field that uses TCNN

    Args:
        aabb: parameters of scene aabb bounds
        num_images: number of images in the dataset
        num_layers: number of hidden layers
        hidden_dim: dimension of hidden layers
        geo_feat_dim: output geo feat dimensions
        num_levels: number of levels of the hashmap for the base mlp
        base_res: base resolution of the hashmap for the base mlp
        max_res: maximum resolution of the hashmap for the base mlp
        log2_hashmap_size: size of the hashmap for the base mlp
        num_layers_color: number of hidden layers for color network
        num_layers_transient: number of hidden layers for transient network
        features_per_level: number of features per level for the hashgrid
        hidden_dim_color: dimension of hidden layers for color network
        hidden_dim_transient: dimension of hidden layers for transient network
        appearance_embedding_dim: dimension of appearance embedding
        transient_embedding_dim: dimension of transient embedding
        use_transient_embedding: whether to use transient embedding
        use_semantics: whether to use semantic segmentation
        num_semantic_classes: number of semantic classes
        use_pred_normals: whether to use predicted normals
        use_average_appearance_embedding: whether to use average appearance embedding or zeros for inference
        spatial_distortion: spatial distortion to apply to the scene
    """

    aabb: Tensor

    def __init__(
        self,
        aabb: Tensor,
        num_images: int,
        grid: int = 3,
        k: int = 3,
        kan_device: str = "cuda",
        num_layers: int = 2,
        hidden_dim: int = 64,
        geo_feat_dim: int = 15,
        num_levels: int = 16,
        base_res: int = 16,
        max_res: int = 2048,
        log2_hashmap_size: int = 19,
        num_layers_color: int = 3,
        num_layers_transient: int = 2,
        features_per_level: int = 2,
        hidden_dim_color: int = 64,
        hidden_dim_transient: int = 64,
        appearance_embedding_dim: int = 32,
        transient_embedding_dim: int = 16,
        use_transient_embedding: bool = False,
        use_semantics: bool = False,
        num_semantic_classes: int = 100,
        pass_semantic_gradients: bool = False,
        use_pred_normals: bool = False,
        use_average_appearance_embedding: bool = False,
        spatial_distortion: Optional[SpatialDistortion] = None,
        implementation: Literal["tcnn", "torch"] = "tcnn",
    ) -> None:
        super().__init__(
            aabb,
            num_images,
            num_layers,
            hidden_dim,
            geo_feat_dim,
            num_levels,
            base_res,
            max_res,
            log2_hashmap_size,
            num_layers_color,
            num_layers_transient,
            features_per_level,
            hidden_dim_color,
            hidden_dim_transient,
            appearance_embedding_dim,
            transient_embedding_dim,
            use_transient_embedding,
            use_semantics,
            num_semantic_classes,
            pass_semantic_gradients,
            use_pred_normals,
            use_average_appearance_embedding,
            spatial_distortion,
            implementation,
        )

        self.mlp_base_mlp = KAN(
            width=[self.mlp_base_grid.get_out_dim()]
            + [hidden_dim] * num_layers
            + [1 + self.geo_feat_dim],
            device=kan_device,
            grid=grid,
            k=k,
            seed=42,
        )
        self.mlp_base[1] = self.mlp_base_mlp

        if self.use_transient_embedding:
            self.mlp_transient = KAN(
                width=[self.geo_feat_dim + self.transient_embedding_dim]
                + [hidden_dim_transient] * num_layers_transient
                +[hidden_dim_transient],
                device=kan_device,
                grid=grid,
                k=k,
                seed=42,
            )

        if self.use_semantics:
            self.mlp_semantics = KAN(
                width=[self.geo_feat_dim]
                + [64, 64]
                + [hidden_dim_transient],
                device=kan_device,
                grid=grid,
                k=k,
                seed=42,
            )

        if self.use_pred_normals:
            self.mlp_pred_normals = KAN(
                width=[self.geo_feat_dim + self.position_encoding.get_out_dim()]
                + [64] * 3
                + [hidden_dim_transient],
                device=kan_device,
                grid=grid,
                k=k,
                seed=42,
            )

        self.mlp_head_base = KAN(
            width=[self.direction_encoding.get_out_dim() + self.geo_feat_dim + self.appearance_embedding_dim]
            + [hidden_dim_color] * num_layers_color
            + [3],
            device=kan_device,
            grid=grid,
            k=k,
            seed=42,
        )

        self.mlp_head = nn.Sequential(self.mlp_head_base, nn.Sigmoid())
