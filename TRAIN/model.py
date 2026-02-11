import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, SiglipVisionModel

FEATURE_DIM = 512
NUM_ANGLES = 9
NUM_JOINTS = 7  # DO NOT CHANGE: This value is intentionally set to 7.

class DINOv3Backbone(nn.Module):
    def __init__(self, model_name, unfreeze_blocks=2):
        super().__init__()
        self.model_name = model_name
        if "siglip" in model_name:
            self.model = SiglipVisionModel.from_pretrained(model_name)
        else:
            self.model = AutoModel.from_pretrained(model_name)

        # Freeze backbone parameters
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Unfreeze last N blocks for fine-tuning
        if unfreeze_blocks > 0:
            if hasattr(self.model, "encoder") and hasattr(self.model.encoder, "layers"):
                # ViT / DINOv2 / SigLIP style
                layers = self.model.encoder.layers
                for i in range(len(layers) - unfreeze_blocks, len(layers)):
                    for param in layers[i].parameters():
                        param.requires_grad = True
            elif hasattr(self.model, "blocks"):
                # Alternative ViT style
                layers = self.model.blocks
                for i in range(len(layers) - unfreeze_blocks, len(layers)):
                    for param in layers[i].parameters():
                        param.requires_grad = True

    def forward(self, image_tensor_batch):
        # Removed torch.no_grad() to allow gradient flow for downstream heads
        if "siglip" in self.model_name:
            outputs = self.model(
                pixel_values=image_tensor_batch,
                interpolate_pos_encoding=True)
            tokens = outputs.last_hidden_state
            patch_tokens = tokens[:, 1:, :]
        else: # DINOv3 계열
            outputs = self.model(pixel_values=image_tensor_batch)
            tokens = outputs.last_hidden_state
            num_reg = int(getattr(self.model.config, "num_register_tokens", 0))
            patch_tokens = tokens[:, 1 + num_reg :, :]
        return patch_tokens

class TokenFuser(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # bias=True를 추가하여 PyTorch CUDA kernel의 gradient stride 문제 완화
        self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        self.refine_blocks = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
    def forward(self, x):
        projected = self.projection(x)
        refined = self.refine_blocks(projected)
        residual = self.residual_conv(x)
        return torch.nn.functional.gelu(refined + residual)

class LightCNNStem(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False), # 1/2
            nn.BatchNorm2d(16),
            nn.GELU()
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False), # 1/4
            nn.BatchNorm2d(32),
            nn.GELU()
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False), # 1/8
            nn.BatchNorm2d(64),
            nn.GELU()
        )
        
    def forward(self, x):
        feat_2 = self.conv_block1(x)  # 1/2
        feat_4 = self.conv_block2(feat_2)  # 1/4
        feat_8 = self.conv_block3(feat_4) # 1/8
        return feat_2, feat_4, feat_8

class FusedUpsampleBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.refine_conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x, skip_feature):
        x = self.upsample(x)
        if x.shape[-2:] != skip_feature.shape[-2:]:
            skip_feature = F.interpolate(
                skip_feature, 
                size=x.shape[-2:], # target H, W
                mode='bilinear', 
                align_corners=False
            )

        fused = torch.cat([x, skip_feature], dim=1)
        return self.refine_conv(fused)

class UNetViTKeypointHead(nn.Module):
    def __init__(self, input_dim=768, num_joints=NUM_JOINTS, heatmap_size=(512, 512)):
        super().__init__()
        self.heatmap_size = heatmap_size
        self.token_fuser = TokenFuser(input_dim, 256)
        self.decoder_block1 = FusedUpsampleBlock(in_channels=256, skip_channels=64, out_channels=128)
        self.decoder_block2 = FusedUpsampleBlock(in_channels=128, skip_channels=32, out_channels=64)
        self.decoder_block3 = FusedUpsampleBlock(in_channels=64, skip_channels=16, out_channels=32)
        self.heatmap_predictor = nn.Conv2d(32, num_joints, kernel_size=3, padding=1)

    def forward(self, dino_features, cnn_features):
        cnn_feat_2, cnn_feat_4, cnn_feat_8 = cnn_features
        b, n, d = dino_features.shape
        h = w = int(math.sqrt(n))

        if h * w != n:
            n_new = h * w
            dino_features = dino_features[:, :n_new, :]
        x = dino_features.permute(0, 2, 1).reshape(b, d, h, w)
        
        x = self.token_fuser(x)
        x = self.decoder_block1(x, cnn_feat_8)
        x = self.decoder_block2(x, cnn_feat_4)
        x = self.decoder_block3(x, cnn_feat_2)
        heatmaps = self.heatmap_predictor(x)
        
        return F.interpolate(heatmaps, size=self.heatmap_size, mode='bilinear', align_corners=False)
        
class JointAngleROIHead(nn.Module):
    """
    Predicts joint angles by pooling backbone features at predicted keypoint locations
    and reasoning about joint relationships using self-attention.
    """
    def __init__(self, input_dim=FEATURE_DIM, num_joints=NUM_JOINTS, num_angles=NUM_ANGLES):
        super().__init__()
        self.num_joints = num_joints
        
        # Per-joint feature refinement
        self.joint_feature_net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.LayerNorm(256)
        )
        
        # Self-attention to reason about joint relationships
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=4,
            dim_feedforward=512,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.joint_relation_net = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Global fusion and angle prediction
        self.angle_predictor = nn.Sequential(
            nn.Linear(256 * num_joints, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, num_angles),
            nn.Tanh() # Scale output to [-1, 1]
        )
    
    def forward(self, dino_features, predicted_heatmaps):
        b, n, d = dino_features.shape
        h = w = int(math.sqrt(n))
        
        # 1. Reshape DINO features to (B, D, H, W)
        feat_map = dino_features.permute(0, 2, 1).reshape(b, d, h, w)
        
        # 2. Weighted pooling of features for each joint (Spatial Softmax approach)
        weights = F.interpolate(predicted_heatmaps, size=(h, w), mode='bilinear', align_corners=False)
        weights = torch.clamp(weights, min=0)
        
        weights_flat = weights.reshape(b, self.num_joints, -1)
        weights_norm = F.softmax(weights_flat / 0.1, dim=-1) # temperature 0.1
        weights_norm = weights_norm.reshape(b, self.num_joints, h, w)
        
        # output: (B, NJ, D)
        joint_features = torch.einsum('bdhw,bjhw->bjd', feat_map, weights_norm)
        
        # 3. Refine and model relations
        refined_features = self.joint_feature_net(joint_features) # (B, NJ, 256)
        related_features = self.joint_relation_net(refined_features) # (B, NJ, 256)
        
        # 4. Predict angles
        flat_features = related_features.reshape(b, -1)
        return self.angle_predictor(flat_features)

class DINOv3PoseEstimator(nn.Module):
    def __init__(self, dino_model_name, heatmap_size, unfreeze_blocks=2):
        super().__init__()
        self.dino_model_name = dino_model_name
        self.backbone = DINOv3Backbone(dino_model_name, unfreeze_blocks=unfreeze_blocks)
        
        if "siglip" in self.dino_model_name:
            config = self.backbone.model.config
            feature_dim = config.hidden_size
        else: # DINOv3 계열
            config = self.backbone.model.config
            feature_dim = config.hidden_sizes[-1] if "conv" in self.dino_model_name else config.hidden_size
        
        self.cnn_stem = LightCNNStem()
        self.keypoint_head = UNetViTKeypointHead(input_dim=feature_dim, heatmap_size=heatmap_size)
        self.angle_head = JointAngleROIHead(input_dim=feature_dim)
        
    def forward(self, image_tensor_batch):
        dino_features = self.backbone(image_tensor_batch)
        cnn_stem_features = self.cnn_stem(image_tensor_batch)
        predicted_heatmaps = self.keypoint_head(dino_features, cnn_stem_features)
        predicted_angles = self.angle_head(dino_features, predicted_heatmaps)
        
        return predicted_heatmaps, predicted_angles
