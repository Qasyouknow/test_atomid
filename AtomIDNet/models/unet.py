from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F


# --- New Residual Block Definition ---
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        main_out = self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x)))))
        shortcut_out = self.shortcut(x)
        final_out = self.relu(main_out + shortcut_out)
        return final_out


# --- Attention Block (Unchanged) ---
class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


# --- Reconstructed UNet class with UNet3+/Attention/ResNet ---
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet, self).__init__()
        features = init_features

        # Encoder
        self.encoder1 = ResidualBlock(in_channels, features)
        self.encoder2 = ResidualBlock(features, features * 2)
        self.encoder3 = ResidualBlock(features * 2, features * 4)
        self.encoder4 = ResidualBlock(features * 4, features * 8)
        self.bottleneck = ResidualBlock(features * 8, features * 16)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Regressor Head
        self.regressor = nn.Sequential(
            nn.Linear(features * 16, features * 8), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(features * 8, features * 2), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(features * 2, 1)
        )

        # --- Decoder Blocks ---
        # Components for Decoder 4
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4_fusion = ResidualBlock(features * 16, features * 8)

        # Components for Decoder 3
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.ag_e2_d3 = Attention_block(F_g=features * 4, F_l=features * 2, F_int=features * 2)
        self.ag_e1_d3 = Attention_block(F_g=features * 4, F_l=features, F_int=features)
        cat3_channels = (features * 4) + (features * 2) + features + (features * 8) + (
                    features * 16)  # E3, E2, E1, D4, BN
        self.decoder3_fusion = ResidualBlock(cat3_channels, features * 4)

        # Components for Decoder 2
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.ag_e1_d2 = Attention_block(F_g=features * 2, F_l=features, F_int=features)
        cat2_channels = (features * 2) + features + (features * 4) + (features * 8) + (
                    features * 16)  # E2, E1, D3, D4, BN
        self.decoder2_fusion = ResidualBlock(cat2_channels, features * 2)

        # Components for Decoder 1
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        cat1_channels = features + (features * 2) + (features * 4) + (features * 8) + (
                    features * 16)  # E1, D2, D3, D4, BN
        self.decoder1_fusion = ResidualBlock(cat1_channels, features)

        # Output layers for Deep Supervision
        self.out_conv1 = nn.Conv2d(features, out_channels, kernel_size=1)
        self.out_conv2 = nn.Conv2d(features * 2, out_channels, kernel_size=1)
        self.out_conv3 = nn.Conv2d(features * 4, out_channels, kernel_size=1)
        self.out_conv4 = nn.Conv2d(features * 8, out_channels, kernel_size=1)
        self.out_conv_bn = nn.Conv2d(features * 16, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))
        bn = self.bottleneck(self.pool(e4))

        # Regressor
        feat = F.adaptive_avg_pool2d(bn, (1, 1)).view(x.size(0), -1)
        diam = self.regressor(feat)

        # --- Decoder ---
        # Decoder 4
        d4_up = self.upconv4(bn)
        d4 = self.decoder4_fusion(torch.cat([d4_up, e4], dim=1))

        # Decoder 3
        d3_target_size = e3.shape[2:]
        d4_to_d3 = F.interpolate(d4, size=d3_target_size, mode='bilinear', align_corners=True)
        bn_to_d3 = F.interpolate(bn, size=d3_target_size, mode='bilinear', align_corners=True)
        gating_signal_d3 = self.upconv3(d4)  # Gating signal from upsampled D4
        e2_to_d3 = F.interpolate(e2, size=d3_target_size, mode='bilinear', align_corners=True)
        e1_to_d3 = F.interpolate(e1, size=d3_target_size, mode='bilinear', align_corners=True)
        e2_att = self.ag_e2_d3(g=gating_signal_d3, x=e2_to_d3)
        e1_att = self.ag_e1_d3(g=gating_signal_d3, x=e1_to_d3)
        cat3 = torch.cat([e3, e2_att, e1_att, d4_to_d3, bn_to_d3], dim=1)
        d3 = self.decoder3_fusion(cat3)

        # Decoder 2
        d2_target_size = e2.shape[2:]
        d3_to_d2 = F.interpolate(d3, size=d2_target_size, mode='bilinear', align_corners=True)
        d4_to_d2 = F.interpolate(d4, size=d2_target_size, mode='bilinear', align_corners=True)
        bn_to_d2 = F.interpolate(bn, size=d2_target_size, mode='bilinear', align_corners=True)
        gating_signal_d2 = self.upconv2(d3)
        e1_to_d2 = F.interpolate(e1, size=d2_target_size, mode='bilinear', align_corners=True)
        e1_att2 = self.ag_e1_d2(g=gating_signal_d2, x=e1_to_d2)
        cat2 = torch.cat([e2, e1_att2, d3_to_d2, d4_to_d2, bn_to_d2], dim=1)
        d2 = self.decoder2_fusion(cat2)

        # Decoder 1
        d1_target_size = e1.shape[2:]
        d2_to_d1 = F.interpolate(d2, size=d1_target_size, mode='bilinear', align_corners=True)
        d3_to_d1 = F.interpolate(d3, size=d1_target_size, mode='bilinear', align_corners=True)
        d4_to_d1 = F.interpolate(d4, size=d1_target_size, mode='bilinear', align_corners=True)
        bn_to_d1 = F.interpolate(bn, size=d1_target_size, mode='bilinear', align_corners=True)
        cat1 = torch.cat([e1, d2_to_d1, d3_to_d1, d4_to_d1, bn_to_d1], dim=1)
        d1 = self.decoder1_fusion(cat1)

        # Deep Supervision Outputs
        out1 = self.out_conv1(d1)
        out2 = self.out_conv2(d2)
        out3 = self.out_conv3(d3)
        out4 = self.out_conv4(d4)
        out_bn = self.out_conv_bn(bn)

        # Upsample all outputs to the size of out1 for consistent loss calculation in trainer
        out2_up = F.interpolate(out2, size=out1.shape[2:], mode='bilinear', align_corners=True)
        out3_up = F.interpolate(out3, size=out1.shape[2:], mode='bilinear', align_corners=True)
        out4_up = F.interpolate(out4, size=out1.shape[2:], mode='bilinear', align_corners=True)
        out_bn_up = F.interpolate(out_bn, size=out1.shape[2:], mode='bilinear', align_corners=True)

        return torch.abs(diam), [torch.sigmoid(o) for o in [out1, out2_up, out3_up, out4_up, out_bn_up]]


def get_model(input_dim=1, output_dim=1, pretrained: bool = True):
    model = UNet(input_dim, output_dim)
    return model