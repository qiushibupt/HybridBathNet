import torch
import torch.nn as nn
import torch.nn.functional as F

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.up(x)
        return x

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)

class FeatureFusionModule(nn.Module):
    def __init__(self, z_channels, w_channels, output_channels):
        super(FeatureFusionModule, self).__init__()

        concat_channels = z_channels + w_channels

        self.fusion_conv1 = nn.Conv2d(concat_channels, output_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.activation = nn.ReLU(inplace=True)
        
        self.fusion_conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(output_channels)
        
        self.residual_proj = None
        if z_channels != output_channels:
            self.residual_proj = nn.Sequential(
                nn.Conv2d(z_channels, output_channels, kernel_size=1),
                nn.BatchNorm2d(output_channels)
            )
    
    def forward(self, f_z, f_w):
        residual = f_z
        if self.residual_proj is not None:
            residual = self.residual_proj(residual)
        
        concat_features = torch.cat([f_z, f_w], dim=1)
        
        fusion = self.fusion_conv1(concat_features)
        fusion = self.bn1(fusion)
        fusion = self.activation(fusion)
        
        fusion = self.fusion_conv2(fusion)
        fusion = self.bn2(fusion)
        
        output = fusion + residual
        output = self.activation(output)
        return output

class WDI_Prediction_Branch(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(WDI_Prediction_Branch, self).__init__()

        filters = [8, 16, 32, 64, 128]

        self.n_channels = n_channels
        self.n_classes = n_classes
        
        self.inc = DoubleConv(n_channels, filters[0])
        self.down1 = Down(filters[0], filters[1])
        self.down2 = Down(filters[1], filters[2])
        self.down3 = Down(filters[2], filters[3])
        
        self.final_conv = nn.Conv2d(filters[3], n_classes, kernel_size=1)
    
    def forward(self, x):

        original_size = x.shape[2:]
        
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        out = self.final_conv(x4)
        
        out = F.interpolate(out, size=original_size, mode='bilinear', align_corners=False)

        return out, [x1, x2, x3, x4]

class HybridBathNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, wdi_channels=3):
        super(HybridBathNet, self).__init__()
        
        filters = [8, 16, 32, 64, 128]
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = conv_block(ch_in=n_channels, ch_out=filters[0])
        self.Conv2 = conv_block(ch_in=filters[0], ch_out=filters[1])
        self.Conv3 = conv_block(ch_in=filters[1], ch_out=filters[2])
        self.Conv4 = conv_block(ch_in=filters[2], ch_out=filters[3])
        self.Conv5 = conv_block(ch_in=filters[3], ch_out=filters[4])
        
        self.Up5 = up_conv(ch_in=filters[4], ch_out=filters[3])
        self.Up_conv5 = conv_block(ch_in=filters[4], ch_out=filters[3])
        self.Up4 = up_conv(ch_in=filters[3], ch_out=filters[2])
        self.Up_conv4 = conv_block(ch_in=filters[3], ch_out=filters[2])
        self.Up3 = up_conv(ch_in=filters[2], ch_out=filters[1])
        self.Up_conv3 = conv_block(ch_in=filters[2], ch_out=filters[1])
        self.Up2 = up_conv(ch_in=filters[1], ch_out=filters[0])
        self.Up_conv2 = conv_block(ch_in=filters[1], ch_out=filters[0])
        self.Conv_1x1 = nn.Conv2d(filters[0], n_classes, kernel_size=1, stride=1, padding=0)
        
        self.wdi_branch = WDI_Prediction_Branch(n_channels=wdi_channels, n_classes=1)
        
        self.ffm1 = FeatureFusionModule(z_channels=filters[0], w_channels=filters[0], output_channels=filters[0])
        self.ffm2 = FeatureFusionModule(z_channels=filters[1], w_channels=filters[1], output_channels=filters[1])
        self.ffm3 = FeatureFusionModule(z_channels=filters[2], w_channels=filters[2], output_channels=filters[2])
        self.ffm4 = FeatureFusionModule(z_channels=filters[3], w_channels=filters[3], output_channels=filters[3])
    
    def forward(self, x, wdi_input=None):
        if wdi_input is None:
            wdi_input = x
            
        wdi_pred, wdi_features = self.wdi_branch(wdi_input)
        
        x1 = self.Conv1(x)

        x1_fused = self.ffm1(x1, wdi_features[0])
        
        x2 = self.Maxpool(x1_fused)
        x2 = self.Conv2(x2)
        x2_fused = self.ffm2(x2, wdi_features[1])
        
        x3 = self.Maxpool(x2_fused)
        x3 = self.Conv3(x3)
        x3_fused = self.ffm3(x3, wdi_features[2])
        
        x4 = self.Maxpool(x3_fused)
        x4 = self.Conv4(x4)
        x4_fused = self.ffm4(x4, wdi_features[3])
        
        x5 = self.Maxpool(x4_fused)
        x5 = self.Conv5(x5)
        
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)
        
        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)
        
        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        
        depth_pred = self.Conv_1x1(d2)
        
        return depth_pred, wdi_pred