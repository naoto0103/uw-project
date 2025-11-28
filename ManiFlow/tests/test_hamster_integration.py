#!/usr/bin/env python3
"""
Integration test for HAMSTER-ManiFlow Phase 3 implementation.

This test verifies:
1. PathTokenEncoder works correctly
2. HAMSTERPathDP3Encoder integrates with ManiFlow
3. ManiFlowTransformerPointcloudPolicy can use HAMSTERPathDP3Encoder
4. End-to-end forward pass works
5. Gradient flow is correct
"""

import sys
import torch
import torch.nn as nn
from types import SimpleNamespace
from termcolor import cprint

# Add ManiFlow to path
sys.path.insert(0, '/home/naoto/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow')

from maniflow.model.vision_3d.hamster_path_encoder import (
    PathTokenEncoder,
    HAMSTERPathDP3Encoder
)
from maniflow.policy.maniflow_pointcloud_policy import ManiFlowTransformerPointcloudPolicy


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


class ShapeMetaEntry:
    """
    Non-dict object for shape_meta entries to avoid recursive dict_apply.
    Mimics OmegaConf behavior where isinstance(obj, dict) returns False.
    """
    def __init__(self, shape, type_str):
        self._shape = shape
        self._type = type_str

    def __getitem__(self, key):
        if key == 'shape':
            return self._shape
        elif key == 'type':
            return self._type
        raise KeyError(key)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


def test_path_token_encoder():
    """Test PathTokenEncoder module."""
    cprint("\n" + "=" * 60, "cyan")
    cprint("Test 1: PathTokenEncoder", "cyan")
    cprint("=" * 60, "cyan")

    encoder = PathTokenEncoder(
        input_dim=3,
        hidden_dim=128,
        output_dim=256,
        num_heads=4,
        num_layers=2,
    )

    batch_size = 4
    max_path_len = 50
    path_data = torch.randn(batch_size, max_path_len, 3)

    # Test forward pass
    output = encoder(path_data)
    assert output.shape == (batch_size, 256), f"Expected (4, 256), got {output.shape}"
    cprint(f"  [PASS] Output shape: {output.shape}", "green")

    # Test with mask
    mask = torch.ones(batch_size, max_path_len, dtype=torch.bool)
    mask[:, 30:] = False
    output_masked = encoder(path_data, mask)
    assert output_masked.shape == (batch_size, 256)
    cprint(f"  [PASS] Masked output shape: {output_masked.shape}", "green")

    # Test gradient flow
    loss = output.sum()
    loss.backward()
    cprint("  [PASS] Gradient flow works", "green")

    cprint("Test 1: PASSED", "green")
    return True


def test_hamster_path_dp3_encoder():
    """Test HAMSTERPathDP3Encoder module."""
    cprint("\n" + "=" * 60, "cyan")
    cprint("Test 2: HAMSTERPathDP3Encoder", "cyan")
    cprint("=" * 60, "cyan")

    pointcloud_encoder_cfg = ConfigDict(
        in_channels=3,
        out_channels=128,
        use_layernorm=True,
        final_norm='layernorm',
        num_points=128,
        pointwise=True
    )

    # observation_space expects shape tuples directly (not nested dicts)
    observation_space = {
        'point_cloud': (1024, 3),
        'agent_pos': (14,),
        'hamster_path': (50, 3),
    }

    encoder = HAMSTERPathDP3Encoder(
        observation_space=observation_space,
        out_channel=128,
        pointcloud_encoder_cfg=pointcloud_encoder_cfg,
        use_pc_color=False,
        pointnet_type='pointnet',
        downsample_points=True,
        path_hidden_dim=128,
        path_output_dim=256,
    )

    batch_size = 4
    observations = {
        'point_cloud': torch.randn(batch_size, 1024, 3),
        'agent_pos': torch.randn(batch_size, 14),
        'hamster_path': torch.randn(batch_size, 50, 3),
    }

    # Test forward pass
    output = encoder(observations)
    expected_dim = 128 + 64 + 256  # pn + state + path = 448
    assert output.shape == (batch_size, 128, expected_dim), f"Expected (4, 128, 448), got {output.shape}"
    cprint(f"  [PASS] Output shape (pointwise): {output.shape}", "green")

    # Test output_shape
    assert encoder.output_shape() == expected_dim
    cprint(f"  [PASS] output_shape() returns {expected_dim}", "green")

    # Test gradient flow
    loss = output.sum()
    loss.backward()
    cprint("  [PASS] Gradient flow works", "green")

    cprint("Test 2: PASSED", "green")
    return True


def test_policy_initialization():
    """Test ManiFlowTransformerPointcloudPolicy with HAMSTERPathDP3Encoder."""
    cprint("\n" + "=" * 60, "cyan")
    cprint("Test 3: Policy Initialization with HAMSTERPathDP3Encoder", "cyan")
    cprint("=" * 60, "cyan")

    # Create shape_meta matching HAMSTER config
    # Use ShapeMetaEntry to avoid dict_apply recursion (mimics OmegaConf)
    shape_meta = {
        'obs': {
            'point_cloud': ShapeMetaEntry((1024, 3), 'point_cloud'),
            'agent_pos': ShapeMetaEntry((14,), 'low_dim'),
            'hamster_path': ShapeMetaEntry((50, 3), 'low_dim'),
        },
        'action': {
            'shape': (14,)
        }
    }

    pointcloud_encoder_cfg = ConfigDict(
        in_channels=3,
        out_channels=128,
        use_layernorm=True,
        final_norm='layernorm',
        num_points=128,
        pointwise=True,
        state_mlp_size=(64, 64)
    )

    # Initialize policy with HAMSTERPathDP3Encoder
    policy = ManiFlowTransformerPointcloudPolicy(
        shape_meta=shape_meta,
        horizon=16,
        n_action_steps=16,
        n_obs_steps=2,
        num_inference_steps=10,
        encoder_type="HAMSTERPathDP3Encoder",
        encoder_output_dim=128,
        pointcloud_encoder_cfg=pointcloud_encoder_cfg,
        visual_cond_len=128,
        n_layer=2,  # Small model for testing
        n_head=4,
        n_emb=256,
        downsample_points=True,
        path_hidden_dim=128,
        path_output_dim=256,
        path_num_heads=4,
        path_num_layers=2,
    )

    # Verify encoder type
    assert policy.encoder_type == "HAMSTERPathDP3Encoder"
    cprint(f"  [PASS] encoder_type: {policy.encoder_type}", "green")

    # Verify obs_feature_dim
    expected_obs_dim = 128 + 64 + 256  # pn + state + path = 448
    assert policy.obs_feature_dim == expected_obs_dim, f"Expected {expected_obs_dim}, got {policy.obs_feature_dim}"
    cprint(f"  [PASS] obs_feature_dim: {policy.obs_feature_dim}", "green")

    # Verify model input dimension
    cprint(f"  [INFO] DiTX cond_dim: {policy.model.vis_cond_obs_emb.in_features}", "yellow")

    cprint("Test 3: PASSED", "green")
    return policy


def test_policy_forward_pass(policy):
    """Test forward pass through the policy."""
    cprint("\n" + "=" * 60, "cyan")
    cprint("Test 4: Policy Forward Pass (Observation Encoding)", "cyan")
    cprint("=" * 60, "cyan")

    batch_size = 2
    n_obs_steps = 2

    # Create dummy observation data
    obs_dict = {
        'point_cloud': torch.randn(batch_size, n_obs_steps, 1024, 3),
        'agent_pos': torch.randn(batch_size, n_obs_steps, 14),
        'hamster_path': torch.randn(batch_size, n_obs_steps, 50, 3),
    }

    # Normalize observations (identity normalizer for testing)
    policy.normalizer.fit(
        data={
            'point_cloud': torch.randn(10, 1024, 3),
            'agent_pos': torch.randn(10, 14),
            'hamster_path': torch.randn(10, 50, 3),
            'action': torch.randn(10, 14),
        },
        last_n_dims=1,
        mode='limits'
    )

    # Test observation encoding
    policy.eval()
    with torch.no_grad():
        nobs = policy.normalizer.normalize(obs_dict)
        nobs['point_cloud'] = nobs['point_cloud'][..., :3]  # Remove color if any

        # Reshape for encoder
        this_nobs = {}
        for key, value in nobs.items():
            # Reshape (B, T, ...) to (B*T, ...)
            this_nobs[key] = value[:, :n_obs_steps, ...].reshape(-1, *value.shape[2:])

        # Encode observations
        nobs_features = policy.obs_encoder(this_nobs)

        # Reshape back
        vis_cond = nobs_features.reshape(batch_size, -1, policy.obs_feature_dim)

    cprint(f"  [INFO] Observation features shape: {nobs_features.shape}", "yellow")
    cprint(f"  [INFO] Visual condition shape: {vis_cond.shape}", "yellow")

    # Expected: (B, n_obs_steps * num_points, obs_feature_dim)
    # With pointwise=True and num_points=128: (2, 2*128, 448)
    expected_shape = (batch_size, n_obs_steps * 128, policy.obs_feature_dim)
    assert vis_cond.shape == expected_shape, f"Expected {expected_shape}, got {vis_cond.shape}"
    cprint(f"  [PASS] Visual condition shape matches expected", "green")

    cprint("Test 4: PASSED", "green")
    return True


def test_compute_loss():
    """Test loss computation with HAMSTER path features."""
    cprint("\n" + "=" * 60, "cyan")
    cprint("Test 5: Loss Computation", "cyan")
    cprint("=" * 60, "cyan")

    # Create a minimal policy
    # Use ShapeMetaEntry to avoid dict_apply recursion (mimics OmegaConf)
    shape_meta = {
        'obs': {
            'point_cloud': ShapeMetaEntry((1024, 3), 'point_cloud'),
            'agent_pos': ShapeMetaEntry((14,), 'low_dim'),
            'hamster_path': ShapeMetaEntry((50, 3), 'low_dim'),
        },
        'action': {'shape': (14,)}
    }

    pointcloud_encoder_cfg = ConfigDict(
        in_channels=3,
        out_channels=128,
        use_layernorm=True,
        final_norm='layernorm',
        num_points=128,
        pointwise=True,
        state_mlp_size=(64, 64)
    )

    policy = ManiFlowTransformerPointcloudPolicy(
        shape_meta=shape_meta,
        horizon=16,
        n_action_steps=16,
        n_obs_steps=2,
        num_inference_steps=10,
        encoder_type="HAMSTERPathDP3Encoder",
        encoder_output_dim=128,
        pointcloud_encoder_cfg=pointcloud_encoder_cfg,
        visual_cond_len=128,
        n_layer=2,
        n_head=4,
        n_emb=256,
        downsample_points=True,
        path_hidden_dim=128,
        path_output_dim=256,
    )

    # Set up normalizer
    policy.normalizer.fit(
        data={
            'point_cloud': torch.randn(10, 1024, 3),
            'agent_pos': torch.randn(10, 14),
            'hamster_path': torch.randn(10, 50, 3),
            'action': torch.randn(10, 14),
        },
        last_n_dims=1,
        mode='limits'
    )

    # Create batch
    batch_size = 4
    batch = {
        'obs': {
            'point_cloud': torch.randn(batch_size, 2, 1024, 3),
            'agent_pos': torch.randn(batch_size, 2, 14),
            'hamster_path': torch.randn(batch_size, 2, 50, 3),
        },
        'action': torch.randn(batch_size, 16, 14),
    }

    # Create EMA model (required for consistency loss)
    from copy import deepcopy
    ema_policy = deepcopy(policy)

    # Compute loss
    policy.train()
    loss, loss_dict = policy.compute_loss(batch, ema_model=ema_policy)

    cprint(f"  [INFO] Total loss: {loss.item():.6f}", "yellow")
    cprint(f"  [INFO] Flow loss: {loss_dict['loss_flow']:.6f}", "yellow")
    cprint(f"  [INFO] Consistency loss: {loss_dict['loss_ct']:.6f}", "yellow")

    # Verify loss is valid
    assert not torch.isnan(loss), "Loss is NaN"
    assert not torch.isinf(loss), "Loss is Inf"
    assert loss.item() > 0, "Loss should be positive"
    cprint(f"  [PASS] Loss is valid", "green")

    # Test backward pass
    loss.backward()
    cprint("  [PASS] Backward pass completed", "green")

    # Verify gradients exist
    has_grad = False
    for name, param in policy.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break

    assert has_grad, "No gradients found in model parameters"
    cprint("  [PASS] Gradients are computed", "green")

    cprint("Test 5: PASSED", "green")
    return True


def main():
    """Run all integration tests."""
    cprint("\n" + "=" * 60, "magenta")
    cprint("HAMSTER-ManiFlow Phase 3 Integration Tests", "magenta")
    cprint("=" * 60, "magenta")

    # Run tests
    try:
        test_path_token_encoder()
        test_hamster_path_dp3_encoder()
        policy = test_policy_initialization()
        test_policy_forward_pass(policy)
        test_compute_loss()

        cprint("\n" + "=" * 60, "green")
        cprint("ALL INTEGRATION TESTS PASSED!", "green")
        cprint("=" * 60, "green")
        cprint("\nPhase 3 Implementation Summary:", "green")
        cprint("  1. PathTokenEncoder: Encodes HAMSTER 2D paths to features", "green")
        cprint("  2. HAMSTERPathDP3Encoder: Integrates path + point cloud + state", "green")
        cprint("  3. Policy: Supports HAMSTERPathDP3Encoder via encoder_type", "green")
        cprint("  4. Config: New YAML files for HAMSTER integration", "green")
        cprint("  5. Loss: Gradient flow works end-to-end", "green")
        cprint("\nNext steps (Phase 4):", "yellow")
        cprint("  - Create HAMSTERManiFlowPolicy (optional)", "yellow")
        cprint("  - Generate HAMSTER paths for real dataset", "yellow")
        cprint("  - Run training with real data", "yellow")
        return True

    except Exception as e:
        cprint(f"\n[FAILED] Test failed with error: {e}", "red")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
