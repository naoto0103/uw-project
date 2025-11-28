"""
HAMSTER Path Encoder Module

This module provides encoders for HAMSTER 2D path information:
- PathTokenEncoder: Encodes variable-length 2D paths into fixed-size feature vectors
- HAMSTERPathDP3Encoder: Extends DP3Encoder to incorporate HAMSTER path features

The path encoder uses attention-based aggregation to handle variable-length paths
and produces a fixed-dimensional feature vector that can be concatenated with
point cloud and state features for the ManiFlow policy.

Reference: HAMSTER (Hierarchical Action Models For Open-World Robot Manipulation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List, Type
from termcolor import cprint

from maniflow.model.vision_3d.pointnet_extractor import (
    PointNetEncoderXYZ,
    PointNetEncoderXYZRGB,
    create_mlp
)
import maniflow.model.vision_3d.point_process as point_process


class PathTokenEncoder(nn.Module):
    """
    Encoder for HAMSTER 2D path sequences.

    Transforms variable-length path sequences into fixed-size feature vectors
    using token embedding and attention-based aggregation.

    Input: (B, M, 3) where M is max_path_length, each point is (x, y, gripper_state)
    Output: (B, output_dim) aggregated path features

    Args:
        input_dim: Dimension of each path point (default: 3 for x, y, gripper)
        hidden_dim: Hidden dimension for token embeddings (default: 128)
        output_dim: Output feature dimension (default: 256)
        num_heads: Number of attention heads (default: 4)
        num_layers: Number of transformer layers (default: 2)
        dropout: Dropout rate (default: 0.1)
        max_path_length: Maximum path length for positional encoding (default: 100)
        use_layernorm: Whether to use layer normalization (default: True)
    """

    def __init__(self,
                 input_dim: int = 3,
                 hidden_dim: int = 128,
                 output_dim: int = 256,
                 num_heads: int = 4,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 max_path_length: int = 100,
                 use_layernorm: bool = True,
                 **kwargs):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_path_length = max_path_length

        cprint(f"[PathTokenEncoder] Initializing with:", "cyan")
        cprint(f"  - input_dim: {input_dim}", "cyan")
        cprint(f"  - hidden_dim: {hidden_dim}", "cyan")
        cprint(f"  - output_dim: {output_dim}", "cyan")
        cprint(f"  - num_heads: {num_heads}", "cyan")
        cprint(f"  - num_layers: {num_layers}", "cyan")
        cprint(f"  - max_path_length: {max_path_length}", "cyan")

        # Token embedding: project path points to hidden dimension
        self.token_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity(),
        )

        # Learnable positional encoding
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, max_path_length, hidden_dim)
        )
        nn.init.normal_(self.pos_embedding, std=0.02)

        # Learnable query token for aggregation (CLS-like token)
        self.query_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.query_token, std=0.02)

        # Transformer encoder layers for self-attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm architecture
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False
        )

        # Final projection to output dimension
        self.output_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity(),
            nn.Linear(hidden_dim, output_dim),
        )

        # Initialize weights
        self._init_weights()

        cprint(f"[PathTokenEncoder] Initialized successfully", "green")

    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self,
                path_tokens: torch.Tensor,
                path_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode path sequences into feature vectors.

        Args:
            path_tokens: (B, M, 3) path point sequences
            path_mask: (B, M) optional boolean mask, True for valid tokens
                      If None, all tokens are considered valid

        Returns:
            path_features: (B, output_dim) aggregated path features
        """
        B, M, D = path_tokens.shape
        device = path_tokens.device

        # Ensure we don't exceed max_path_length
        if M > self.max_path_length:
            cprint(f"[PathTokenEncoder] Warning: path length {M} exceeds max {self.max_path_length}, truncating", "yellow")
            path_tokens = path_tokens[:, :self.max_path_length, :]
            M = self.max_path_length
            if path_mask is not None:
                path_mask = path_mask[:, :self.max_path_length]

        # 1. Token embedding
        token_emb = self.token_embedding(path_tokens)  # (B, M, hidden_dim)

        # 2. Add positional encoding
        token_emb = token_emb + self.pos_embedding[:, :M, :]  # (B, M, hidden_dim)

        # 3. Prepend query token
        query_tokens = self.query_token.expand(B, -1, -1)  # (B, 1, hidden_dim)
        tokens = torch.cat([query_tokens, token_emb], dim=1)  # (B, M+1, hidden_dim)

        # 4. Create attention mask for transformer
        # Transformer expects: True = masked (ignored), False = attended
        if path_mask is not None:
            # Prepend False for query token (always attend to it)
            query_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)
            src_key_padding_mask = torch.cat([query_mask, ~path_mask], dim=1)  # (B, M+1)
        else:
            src_key_padding_mask = None

        # 5. Apply transformer layers
        transformed = self.transformer(
            tokens,
            src_key_padding_mask=src_key_padding_mask
        )  # (B, M+1, hidden_dim)

        # 6. Extract query token output (aggregated representation)
        aggregated = transformed[:, 0, :]  # (B, hidden_dim)

        # 7. Project to output dimension
        path_features = self.output_projection(aggregated)  # (B, output_dim)

        return path_features

    def compute_path_mask(self, path_tokens: torch.Tensor) -> torch.Tensor:
        """
        Compute a mask for valid path points.

        Points are considered invalid if they are all zeros (padding).

        Args:
            path_tokens: (B, M, 3) path sequences

        Returns:
            mask: (B, M) boolean mask, True for valid points
        """
        # A point is valid if not all coordinates are zero
        # This handles the case where paths are zero-padded
        return ~torch.all(path_tokens == 0, dim=-1)


class HAMSTERPathDP3Encoder(nn.Module):
    """
    DP3Encoder extended with HAMSTER path information.

    This encoder combines:
    1. Point cloud features (via PointNet)
    2. Robot state features (via MLP)
    3. HAMSTER path features (via PathTokenEncoder)

    The final output is a concatenation of these three feature streams.

    Args:
        observation_space: Dict mapping observation keys to their shapes
            Expected keys: 'point_cloud', 'agent_pos', 'hamster_path'
        out_channel: Output channel for point cloud encoder (default: 256)
        state_mlp_size: MLP architecture for state encoding (default: (64, 64))
        state_mlp_activation_fn: Activation function for state MLP
        pointcloud_encoder_cfg: Configuration for point cloud encoder
        use_pc_color: Whether point cloud includes color (default: False)
        pointnet_type: Type of pointnet encoder (default: 'pointnet')
        downsample_points: Whether to downsample points (default: False)
        path_hidden_dim: Hidden dimension for path encoder (default: 128)
        path_output_dim: Output dimension for path encoder (default: 256)
        path_num_heads: Number of attention heads in path encoder (default: 4)
        path_num_layers: Number of transformer layers in path encoder (default: 2)
    """

    def __init__(self,
                 observation_space: Dict,
                 img_crop_shape=None,
                 out_channel: int = 256,
                 state_mlp_size: Tuple[int, ...] = (64, 64),
                 state_mlp_activation_fn: Type[nn.Module] = nn.ReLU,
                 pointcloud_encoder_cfg=None,
                 use_pc_color: bool = False,
                 pointnet_type: str = 'pointnet',
                 downsample_points: bool = False,
                 # Path encoder parameters
                 path_hidden_dim: int = 128,
                 path_output_dim: int = 256,
                 path_num_heads: int = 4,
                 path_num_layers: int = 2,
                 **kwargs):
        super().__init__()

        # Keys for different observation modalities
        self.imagination_key = 'imagin_robot'
        self.state_key = 'agent_pos'
        self.point_cloud_key = 'point_cloud'
        self.path_key = 'hamster_path'

        # Parse observation space
        self.point_cloud_shape = observation_space[self.point_cloud_key]
        self.state_shape = observation_space[self.state_key]

        # Check if hamster_path is in observation space
        if self.path_key not in observation_space:
            raise ValueError(
                f"HAMSTERPathDP3Encoder requires '{self.path_key}' in observation_space. "
                f"Available keys: {list(observation_space.keys())}"
            )
        self.path_shape = observation_space[self.path_key]

        # Check for imagined robot points
        self.use_imagined_robot = self.imagination_key in observation_space
        if self.use_imagined_robot:
            self.imagination_shape = observation_space[self.imagination_key]
        else:
            self.imagination_shape = None

        # Get state MLP size from config if available
        if pointcloud_encoder_cfg is not None:
            if hasattr(pointcloud_encoder_cfg, 'get'):
                state_mlp_size = pointcloud_encoder_cfg.get('state_mlp_size', state_mlp_size)
            elif hasattr(pointcloud_encoder_cfg, 'state_mlp_size'):
                state_mlp_size = pointcloud_encoder_cfg.state_mlp_size

        # Setup point downsampling
        self.downsample_points = downsample_points
        if self.downsample_points:
            self.point_preprocess = point_process.fps_torch
            self.num_points = pointcloud_encoder_cfg.num_points
        else:
            self.point_preprocess = nn.Identity()
            self.num_points = self.point_cloud_shape[0] if isinstance(self.point_cloud_shape, tuple) else self.point_cloud_shape

        cprint(f"[HAMSTERPathDP3Encoder] Initializing:", "yellow")
        cprint(f"  - point_cloud_shape: {self.point_cloud_shape}", "yellow")
        cprint(f"  - state_shape: {self.state_shape}", "yellow")
        cprint(f"  - path_shape: {self.path_shape}", "yellow")
        cprint(f"  - imagination_shape: {self.imagination_shape}", "yellow")
        cprint(f"  - state_mlp_size: {state_mlp_size}", "yellow")
        if self.downsample_points:
            cprint(f"  - downsampling to {self.num_points} points", "yellow")

        # 1. Point cloud encoder (same as DP3Encoder)
        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type

        if pointnet_type == "pointnet":
            if use_pc_color:
                pointcloud_encoder_cfg.in_channels = 6
                self.extractor = PointNetEncoderXYZRGB(**pointcloud_encoder_cfg)
            else:
                pointcloud_encoder_cfg.in_channels = 3
                self.extractor = PointNetEncoderXYZ(**pointcloud_encoder_cfg)
        else:
            raise NotImplementedError(f"pointnet_type: {pointnet_type}")

        # 2. State MLP (same as DP3Encoder)
        if len(state_mlp_size) == 0:
            raise RuntimeError("State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = list(state_mlp_size[:-1])
        state_output_dim = state_mlp_size[-1]

        state_input_dim = self.state_shape[0] if isinstance(self.state_shape, tuple) else self.state_shape
        self.state_mlp = nn.Sequential(
            *create_mlp(state_input_dim, state_output_dim, net_arch, state_mlp_activation_fn)
        )

        # 3. Path encoder (NEW)
        path_input_dim = self.path_shape[-1] if isinstance(self.path_shape, tuple) else 3
        max_path_len = self.path_shape[0] if isinstance(self.path_shape, tuple) else 50

        self.path_encoder = PathTokenEncoder(
            input_dim=path_input_dim,
            hidden_dim=path_hidden_dim,
            output_dim=path_output_dim,
            num_heads=path_num_heads,
            num_layers=path_num_layers,
            max_path_length=max_path_len,
        )

        # 4. Calculate total output dimension
        # Point cloud features + State features + Path features
        self.n_output_channels = out_channel + state_output_dim + path_output_dim
        self.pointwise = pointcloud_encoder_cfg.get('pointwise', False) if hasattr(pointcloud_encoder_cfg, 'get') else getattr(pointcloud_encoder_cfg, 'pointwise', False)

        cprint(f"[HAMSTERPathDP3Encoder] Output composition:", "red")
        cprint(f"  - Point cloud features: {out_channel}", "red")
        cprint(f"  - State features: {state_output_dim}", "red")
        cprint(f"  - Path features: {path_output_dim}", "red")
        cprint(f"  - Total output dim: {self.n_output_channels}", "red")
        cprint(f"  - Pointwise mode: {self.pointwise}", "red")
        if self.pointwise:
            cprint(f"  - Output points num: {self.num_points}", "red")
        else:
            cprint(f"  - Output points num: 1 (global feature)", "red")

    def forward(self, observations: Dict) -> torch.Tensor:
        """
        Encode observations into feature vectors.

        Args:
            observations: Dict containing:
                - 'point_cloud': (B, N, C) point cloud data
                - 'agent_pos': (B, D_state) robot state
                - 'hamster_path': (B, M, 3) HAMSTER path sequence

        Returns:
            features: (B, n_output_channels) or (B, N, n_output_channels) if pointwise
        """
        # 1. Process point cloud (same as DP3Encoder)
        points = observations[self.point_cloud_key]
        assert len(points.shape) == 3, f"point cloud shape: {points.shape}, expected 3D tensor"

        # Concatenate imagined robot points if available
        if self.use_imagined_robot:
            img_points = observations[self.imagination_key][..., :points.shape[-1]]
            points = torch.cat([points, img_points], dim=1)

        # Downsample if needed
        if self.downsample_points and points.shape[1] > self.num_points:
            points, _ = self.point_preprocess(points, self.num_points)

        # Encode point cloud
        pn_feat = self.extractor(points)  # (B, out_channel) or (B, N, out_channel)

        # 2. Process state (same as DP3Encoder)
        state = observations[self.state_key]
        state_feat = self.state_mlp(state)  # (B, state_output_dim)

        # 3. Process HAMSTER path (NEW)
        path_tokens = observations[self.path_key]  # (B, M, 3)

        # Compute mask for valid path points (non-zero)
        path_mask = self.path_encoder.compute_path_mask(path_tokens)

        # Encode path
        path_feat = self.path_encoder(path_tokens, path_mask)  # (B, path_output_dim)

        # 4. Concatenate features
        if len(pn_feat.shape) == 3:
            # Pointwise mode: broadcast state and path features to each point
            # pn_feat: (B, N, pn_dim)
            # state_feat: (B, state_dim) -> (B, N, state_dim)
            # path_feat: (B, path_dim) -> (B, N, path_dim)
            N = pn_feat.shape[1]
            state_feat_expanded = state_feat.unsqueeze(1).expand(-1, N, -1)
            path_feat_expanded = path_feat.unsqueeze(1).expand(-1, N, -1)
            final_feat = torch.cat([pn_feat, state_feat_expanded, path_feat_expanded], dim=-1)
        else:
            # Global feature mode: simple concatenation
            # pn_feat: (B, pn_dim)
            # state_feat: (B, state_dim)
            # path_feat: (B, path_dim)
            final_feat = torch.cat([pn_feat, state_feat, path_feat], dim=-1)

        return final_feat

    def output_shape(self) -> int:
        """Return the output feature dimension."""
        return self.n_output_channels


if __name__ == "__main__":
    """Test the HAMSTER path encoders."""
    import torch

    print("=" * 60)
    print("Testing PathTokenEncoder")
    print("=" * 60)

    # Test PathTokenEncoder
    batch_size = 4
    max_path_len = 50
    path_dim = 3

    path_encoder = PathTokenEncoder(
        input_dim=path_dim,
        hidden_dim=128,
        output_dim=256,
        num_heads=4,
        num_layers=2,
    )

    # Create dummy path data
    path_data = torch.randn(batch_size, max_path_len, path_dim)

    # Forward pass
    path_features = path_encoder(path_data)
    print(f"\nPathTokenEncoder test:")
    print(f"  Input shape: {path_data.shape}")
    print(f"  Output shape: {path_features.shape}")
    assert path_features.shape == (batch_size, 256), f"Expected {(batch_size, 256)}, got {path_features.shape}"
    print("  [PASS] Output shape is correct")

    # Test with padding mask
    path_mask = torch.ones(batch_size, max_path_len, dtype=torch.bool)
    path_mask[:, 30:] = False  # Last 20 points are padding

    path_features_masked = path_encoder(path_data, path_mask)
    print(f"  With mask - Output shape: {path_features_masked.shape}")
    assert path_features_masked.shape == (batch_size, 256)
    print("  [PASS] Masked output shape is correct")

    # Test gradient flow
    loss = path_features.sum()
    loss.backward()
    print("  [PASS] Gradient flow is working")

    print("\n" + "=" * 60)
    print("Testing HAMSTERPathDP3Encoder")
    print("=" * 60)

    # Test HAMSTERPathDP3Encoder
    # Use a simple dict-like object for testing
    class ConfigDict(dict):
        """Dict that also supports attribute access."""
        def __getattr__(self, key):
            try:
                return self[key]
            except KeyError:
                raise AttributeError(key)

        def __setattr__(self, key, value):
            self[key] = value

        def get(self, key, default=None):
            return super().get(key, default)

    pointcloud_encoder_cfg = ConfigDict(
        in_channels=3,
        out_channels=128,
        use_layernorm=True,
        final_norm='layernorm',
        num_points=128,
        pointwise=True
    )

    observation_space = {
        'point_cloud': (1024, 3),
        'agent_pos': (14,),
        'hamster_path': (50, 3),
    }

    hamster_encoder = HAMSTERPathDP3Encoder(
        observation_space=observation_space,
        out_channel=128,
        state_mlp_size=(64, 64),
        pointcloud_encoder_cfg=pointcloud_encoder_cfg,
        use_pc_color=False,
        pointnet_type='pointnet',
        downsample_points=True,
        path_hidden_dim=128,
        path_output_dim=256,
        path_num_heads=4,
        path_num_layers=2,
    )

    # Create dummy observations
    observations = {
        'point_cloud': torch.randn(batch_size, 1024, 3),
        'agent_pos': torch.randn(batch_size, 14),
        'hamster_path': torch.randn(batch_size, 50, 3),
    }

    # Forward pass
    features = hamster_encoder(observations)
    expected_dim = 128 + 64 + 256  # pn_out + state_out + path_out = 448

    print(f"\nHAMSTERPathDP3Encoder test (pointwise=True):")
    print(f"  Point cloud input: {observations['point_cloud'].shape}")
    print(f"  Agent pos input: {observations['agent_pos'].shape}")
    print(f"  HAMSTER path input: {observations['hamster_path'].shape}")
    print(f"  Output shape: {features.shape}")
    print(f"  Expected output dim: {expected_dim}")

    assert features.shape == (batch_size, 128, expected_dim), f"Expected {(batch_size, 128, expected_dim)}, got {features.shape}"
    print(f"  [PASS] Output shape is correct (B, N, {expected_dim})")

    assert hamster_encoder.output_shape() == expected_dim
    print(f"  [PASS] output_shape() returns {expected_dim}")

    # Test gradient flow
    loss = features.sum()
    loss.backward()
    print("  [PASS] Gradient flow is working")

    # Test with pointwise=False (global features)
    print("\nTesting with pointwise=False:")
    pointcloud_encoder_cfg.pointwise = False

    hamster_encoder_global = HAMSTERPathDP3Encoder(
        observation_space=observation_space,
        out_channel=128,
        state_mlp_size=(64, 64),
        pointcloud_encoder_cfg=pointcloud_encoder_cfg,
        use_pc_color=False,
        pointnet_type='pointnet',
        downsample_points=False,
        path_hidden_dim=128,
        path_output_dim=256,
    )

    features_global = hamster_encoder_global(observations)
    print(f"  Output shape (global): {features_global.shape}")
    assert features_global.shape == (batch_size, expected_dim), f"Expected {(batch_size, expected_dim)}, got {features_global.shape}"
    print(f"  [PASS] Global feature output shape is correct")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
