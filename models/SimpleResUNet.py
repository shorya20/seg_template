import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.layers.factories import Act, Norm

class SimpleResBlock(nn.Module):
    """
    Simplified Residual Block: Conv -> BN -> ReLU -> Conv -> BN + skip -> ReLU
    """
    def __init__(self, in_channels, out_channels, stride=1, norm=Norm.INSTANCE, act=Act.RELU, dropout=0.0, spatial_dims=3):
        super().__init__()
        
        # Main path
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = Norm[norm.value if hasattr(norm, 'value') else norm, spatial_dims](out_channels)
        self.act1 = Act[act.value if hasattr(act, 'value') else act](inplace=True)
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = Norm[norm.value if hasattr(norm, 'value') else norm, spatial_dims](out_channels)
        
        # Skip connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                Norm[norm.value if hasattr(norm, 'value') else norm, spatial_dims](out_channels)
            )
        else:
            self.shortcut = nn.Identity()
            
        self.act_final = Act[act.value if hasattr(act, 'value') else act](inplace=True)
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        
        out += identity
        out = self.act_final(out)
        
        return out

class SimpleResUNet(nn.Module):
    """
    Simplified 3D Residual U-Net without attention gates
    """
    def __init__(self, spatial_dims=3, in_channels=1, out_channels=1, channels=32, 
                 strides=2, norm=Norm.INSTANCE, act=Act.RELU, dropout=0.1):
        super().__init__()
        
        if spatial_dims != 3:
            raise ValueError("This SimpleResUNet implementation currently only supports spatial_dims=3.")
            
        # Channel progression: [32, 64, 128, 256, 512]
        f = [channels * (2**i) for i in range(5)]
        
        # Initial convolution
        self.initial_conv = nn.Conv3d(in_channels, f[0], kernel_size=3, padding=1, bias=False)
        self.initial_norm = Norm[norm.value if hasattr(norm, 'value') else norm, spatial_dims](f[0])
        self.initial_act = Act[act.value if hasattr(act, 'value') else act](inplace=True)
        
        # Encoder
        self.enc1 = SimpleResBlock(f[0], f[0], stride=1, norm=norm, act=act, dropout=dropout, spatial_dims=spatial_dims)
        self.enc2 = SimpleResBlock(f[0], f[1], stride=2, norm=norm, act=act, dropout=dropout, spatial_dims=spatial_dims)
        self.enc3 = SimpleResBlock(f[1], f[2], stride=2, norm=norm, act=act, dropout=dropout, spatial_dims=spatial_dims)
        self.enc4 = SimpleResBlock(f[2], f[3], stride=2, norm=norm, act=act, dropout=dropout, spatial_dims=spatial_dims)
        
        # Bottleneck
        self.bottleneck = SimpleResBlock(f[3], f[4], stride=2, norm=norm, act=act, dropout=dropout, spatial_dims=spatial_dims)
        
        # Decoder upsampling
        self.up4 = nn.ConvTranspose3d(f[4], f[3], kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose3d(f[3], f[2], kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose3d(f[2], f[1], kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose3d(f[1], f[0], kernel_size=2, stride=2)
        
        # Decoder blocks
        self.dec4 = SimpleResBlock(f[3] + f[3], f[3], stride=1, norm=norm, act=act, dropout=dropout, spatial_dims=spatial_dims)
        self.dec3 = SimpleResBlock(f[2] + f[2], f[2], stride=1, norm=norm, act=act, dropout=dropout, spatial_dims=spatial_dims)
        self.dec2 = SimpleResBlock(f[1] + f[1], f[1], stride=1, norm=norm, act=act, dropout=dropout, spatial_dims=spatial_dims)
        self.dec1 = SimpleResBlock(f[0] + f[0], f[0], stride=1, norm=norm, act=act, dropout=dropout, spatial_dims=spatial_dims)
        
        # Final output
        self.final_conv = nn.Conv3d(f[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Initial conv
        x = self.initial_conv(x)
        x = self.initial_norm(x)
        x = self.initial_act(x)
        
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        
        # Bottleneck
        b = self.bottleneck(e4)
        
        # Decoder
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        # Final output
        output = self.final_conv(d1)
        return output 