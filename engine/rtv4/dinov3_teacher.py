"""
RT-DETRv4: Painlessly Furthering Real-Time Object Detection with Vision Foundation Models
Copyright (c) 2025 The RT-DETRv4 Authors. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core import register
import logging
from pathlib import Path
from torchvision.transforms import v2 as transforms

_logger = logging.getLogger(__name__)


@register()
class DINOv3TeacherModel(nn.Module):
    def __init__(self,
                 dinov3_repo_path: str,
                 dinov3_weights_path: str,
                 dinov3_model_type: str = "dinov3_vitb16",
                 patch_size: int = 16,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)):
        super().__init__()
        self.dinov3_repo_path = dinov3_repo_path
        self.dinov3_weights_path = dinov3_weights_path
        self.patch_size = patch_size

        repo_path = Path(dinov3_repo_path).expanduser().resolve()
        weights_path = Path(dinov3_weights_path).expanduser().resolve()
        hubconf_path = repo_path / 'hubconf.py'
        if not hubconf_path.is_file():
            raise FileNotFoundError(
                'DINOv3 repository is not installed at the configured path. '
                f'Expected: {hubconf_path}\n'
                'Clone the official DINOv3 repository there, or set '
                '`teacher_model.dinov3_repo_path` to the directory that contains '
                '`hubconf.py`.')
        if not weights_path.is_file():
            raise FileNotFoundError(
                'DINOv3 teacher checkpoint was not found. '
                f'Expected: {weights_path}\n'
                'Download the checkpoint and set '
                '`teacher_model.dinov3_weights_path` to its file path.')
        self.dinov3_repo_path = str(repo_path)
        self.dinov3_weights_path = str(weights_path)

        _logger.info(f"[Teacher Model] Attempting to load DINOv3 teacher via torch.hub.load...")
        _logger.info(f"[Teacher Model] DINOv3 repo path: {self.dinov3_repo_path}")
        _logger.info(f"[Teacher Model] DINOv3 weights path: {self.dinov3_weights_path}")

        try:
            self.model = torch.hub.load(
                self.dinov3_repo_path,
                dinov3_model_type,
                source='local',
                weights=self.dinov3_weights_path
            )
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

            _logger.info(f"[Teacher Model] Successfully loaded DINOv3 teacher from local repo and weights.")
            self.teacher_feature_dim = self.model.embed_dim

        except Exception as e:
            _logger.error(f"[Teacher Model] Failed to load DINOv3: {e}")
            raise

        self.normalize_transform = transforms.Normalize(mean=mean, std=std)
        self.avgpool_2x2 = nn.AvgPool2d(kernel_size=2, stride=2)

        _logger.info(f"[Teacher Model] DINOv3 initialized. Feature dimension: {self.teacher_feature_dim}.")
        _logger.info(
            f"[Teacher Model] Teacher model is configured to output features at a resolution that is 2x2 of the student's highest-level features after 2x downsampling.")

    def forward(self, images: torch.Tensor):
        processed_images = self.avgpool_2x2(self.normalize_transform(images))

        with torch.no_grad():
            dinov3_output_dict = self.model(processed_images, is_training=True, masks=None)
            patch_tokens = dinov3_output_dict["x_norm_patchtokens"]

            if patch_tokens.dim() != 3:
                _logger.error(
                    f"[Teacher Model] Expected 3D patch tokens, but got {patch_tokens.dim()}D tensor. Shape: {patch_tokens.shape}")
                raise ValueError("DINOv3 model's output 'x_norm_patchtokens' is not in expected 3D format.")

            B, N_patches, C_teacher = patch_tokens.shape

            H_patches_out = processed_images.shape[-2] // self.patch_size
            W_patches_out = processed_images.shape[-1] // self.patch_size
            if H_patches_out * W_patches_out != N_patches:
                _logger.error(
                    f"[Teacher Model] Number of patches {N_patches} is not a perfect square for spatial reshape. Input image size: {images.shape[2:]}. Patch size: {self.patch_size}.")
                raise ValueError(
                    f"Number of patches {N_patches} does not match the expected "
                    f"{H_patches_out}x{W_patches_out} grid. Check the input size "
                    "and configured patch_size.")

            teacher_feature_map = patch_tokens.permute(0, 2, 1).reshape(B, C_teacher, H_patches_out, W_patches_out)

            _logger.info(
                f"[Teacher Model] Spatial size: {teacher_feature_map.shape[2:]}")

            return teacher_feature_map.detach()
