#!/usr/bin/python3
'''
This work is published in MobiCom 2024 Hydra: Accurate Multi-Modal Leaf Wetness Sensing with mm-Wave and Camera
This work is published under the MIT License.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

# Multi-Modal Fusion Layer
class MaskScalingLayer(nn.Module):
    def __init__(self):
        super(MaskScalingLayer, self).__init__()
        self.dense = nn.Linear(224 * 224, 1)  # Linear layer to learn the scaling factor

    def forward(self, mask, image):
        flattened_mask = mask.view(mask.size(0), -1)  # Flatten the mask
        scaling_factor = self.dense(flattened_mask)  # Scale the flattened mask
        scaling_factor = scaling_factor.view(-1, 1, 1, 1)  # Reshape to match the dimensions
        scaled_mask = mask * scaling_factor
        return image * scaled_mask

# Residual Block for ResNet-18 Feature Extraction
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.relu = nn.ReLU()

    def forward(self, x):
        shortcut = self.shortcut(x)  # Shortcut path
        x = self.conv1(x)  # First convolution
        x = self.relu(x)
        x = self.conv2(x)  # Second convolution
        x += shortcut  # Skip connection
        x = self.relu(x)
        return x

# ResNet-18 Feature Extractor
class ResNet18FeatureExtractor(nn.Module):
    def __init__(self):
        super(ResNet18FeatureExtractor, self).__init__()
        self.layer1 = self._make_layer(3, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

# Transformer Encoder Block
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = x + self.dropout(attn_output)  # Add & Norm
        x = self.layernorm1(x)
        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)  # Add & Norm
        x = self.layernorm2(x)
        return x

# Depth-aware Positional Encoding for Transformer
class DepthAwarePositionalEncoding(nn.Module):
    def __init__(self, depth, feature_dim):
        super(DepthAwarePositionalEncoding, self).__init__()
        self.pos_encoding = self.create_positional_encoding(depth, feature_dim)

    def create_positional_encoding(self, depth, feature_dim):
        position = torch.arange(depth, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, feature_dim, 2).float() * -(torch.log(torch.tensor(10000.0)) / feature_dim))
        pe = torch.zeros(depth, feature_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pos_encoding[:, :x.size(1), :]

# Single Depth Feature Extraction for SAR and RGB fusion
class SingleDepthFeatureExtractor(nn.Module):
    def __init__(self):
        super(SingleDepthFeatureExtractor, self).__init__()
        self.mask_scaling_layer = MaskScalingLayer()
        self.feature_extractor = ResNet18FeatureExtractor()

    def forward(self, sar_input, image_input):
        fused_feature = self.mask_scaling_layer(sar_input, image_input)
        extracted_feature = self.feature_extractor(fused_feature)
        return F.adaptive_avg_pool2d(extracted_feature, (1, 1)).squeeze()

# Hydra Model
class HydraModel(nn.Module):
    def __init__(self, sar_shape=(224, 224, 10), image_shape=(224, 224, 1)):
        super(HydraModel, self).__init__()
        self.sar_shape = sar_shape
        self.image_shape = image_shape

        # Process each SAR channel independently
        self.single_depth_fe = SingleDepthFeatureExtractor()
        self.positional_encoding = DepthAwarePositionalEncoding(depth=sar_shape[2], feature_dim=512)
        self.transformer_block = TransformerEncoder(embed_dim=512, num_heads=8, ff_dim=2048)
        self.fc = nn.Linear(512, 1)  # Final classification layer

    def forward(self, sar_input, image_input):
        fused_features = []
        for i in range(self.sar_shape[2]):
            fused_features.append(self.single_depth_fe(sar_input[:, :, :, i:i+1], image_input))
        fused_features = torch.stack(fused_features, dim=1)

        x = self.positional_encoding(fused_features)
        x = x.transpose(0, 1)  # Transformer expects (sequence, batch, feature)
        x = self.transformer_block(x)
        x = x.mean(dim=0)  # Global Average Pooling (across sequence dimension)

        output = torch.sigmoid(self.fc(x))  # Binary classification
        return output

# Example usage
if __name__ == "__main__":
    sar_shape = (224, 224, 10)
    image_shape = (224, 224, 1)

    model = HydraModel(sar_shape, image_shape)

    print(model)
