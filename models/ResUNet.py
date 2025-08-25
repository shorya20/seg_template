import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.layers.factories import Act, Norm

# Reference: Inspired by the ResUnet-a architecture 
# https://github.com/Akhilesh64/ResUnet-a/blob/main/model.py
# Adapted for 3D and integrated into a PyTorch/MONAI-like framework.

class ConvBlock(nn.Module):
    """
    Convolutional Block: Conv3d -> Norm -> Act
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, norm=Norm.INSTANCE, act=Act.RELU, dropout=0.0, spatial_dims=3):
        super().__init__()
        padding = kernel_size // 2
        #MONAI 1.4.0+ requires Norm -> Act -> Conv (pre-activation style). Norm factory expects (spatial_dims, in_features)
        layers = []
        if isinstance(norm, str) or hasattr(norm, 'value'):
            norm_str = norm.value if hasattr(norm, 'value') else norm
            layers.append(Norm[norm_str, spatial_dims](in_channels))
        else:
            layers.append(norm(in_channels))
        
        #Handle activation factory call
        if isinstance(act, str) or hasattr(act, 'value'):
            act_str = act.value if hasattr(act, 'value') else act
            layers.append(Act[act_str](inplace=True))
        else:
            layers.append(act(inplace=True))

        #Add conv3d
        layers.append(nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False))
        
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)

class ResidualConvBlock(nn.Module):
    """
    Residual Convolutional Block with optional dropout
    """
    def __init__(self, in_channels, out_channels, stride=1, norm=Norm.INSTANCE, act=Act.RELU, dropout=0.0, spatial_dims=3):
        super().__init__()

        self.conv_block1 = ConvBlock(in_channels, out_channels, kernel_size=3, stride=stride, norm=norm, act=act, dropout=dropout, spatial_dims=spatial_dims)
        self.conv_block2 = ConvBlock(out_channels, out_channels, kernel_size=3, stride=1, norm=norm, act=act, dropout=0.0, spatial_dims=spatial_dims)
        # Shortcut connection matching dimensions
        if stride != 1 or in_channels != out_channels:
            shortcut_layers = []
            # Handle norm factory call properly for shortcut
            if isinstance(norm, str) or hasattr(norm, 'value'):
                norm_str = norm.value if hasattr(norm, 'value') else norm
                shortcut_layers.append(Norm[norm_str, spatial_dims](in_channels))
            else:
                shortcut_layers.append(norm(in_channels))
            
            # Handle activation factory call properly for shortcut
            if isinstance(act, str) or hasattr(act, 'value'):
                act_str = act.value if hasattr(act, 'value') else act
                shortcut_layers.append(Act[act_str](inplace=True))
            else:
                shortcut_layers.append(act(inplace=True))
            
            shortcut_layers.append(nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False))
            self.shortcut = nn.Sequential(*shortcut_layers)
        else:
            self.shortcut = nn.Identity()
            
        if isinstance(act, str) or hasattr(act, 'value'):
            act_str = act.value if hasattr(act, 'value') else act
            self.act_out = Act[act_str](inplace=True)
        else:
            self.act_out = act(inplace=True)

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x + shortcut
        # x = self.act_out(x) # Optional: Might be redundant if next layer starts with Norm->Act
        return x

class AttentionGate(nn.Module):
    """
    Attention Gate mechanism for U-Net skip connections
    """
    def __init__(self, in_channels_g, in_channels_x, inter_channels, norm=Norm.INSTANCE, act=Act.RELU, spatial_dims=3):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(in_channels_g, inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
        )
        
        self.W_x = nn.Sequential(
            nn.Conv3d(in_channels_x, inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
        )
        
        self.psi = nn.Sequential(
            nn.Conv3d(inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )
        
        # activation factory call properly
        if isinstance(act, str) or hasattr(act, 'value'):
            act_str = act.value if hasattr(act, 'value') else act
            self.relu = Act[act_str](inplace=True)
        else:
            self.relu = act(inplace=True)

    def forward(self, g, x):
        """
        Args:
            g: Gating signal (from deeper layer)
            x: Skip connection signal (from corresponding encoder layer)
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi_in = self.relu(g1 + x1)
        psi_out = self.psi(psi_in)
        return x * psi_out

class EncoderBlock(nn.Module):
    """
    Encoder Block: ResidualConvBlock -> MaxPool
    """
    def __init__(self, in_channels, out_channels, pool_kernel_size=2, pool_stride=2, norm=Norm.INSTANCE, act=Act.RELU, dropout=0.0, spatial_dims=3):
        super().__init__()
        self.res_block = ResidualConvBlock(in_channels, out_channels, stride=1, norm=norm, act=act, dropout=dropout, spatial_dims=spatial_dims)
        self.pool = nn.MaxPool3d(kernel_size=pool_kernel_size, stride=pool_stride)

    def forward(self, x):
        s = self.res_block(x) # Output for skip connection
        p = self.pool(s)      # Output for next layer
        return s, p

class DecoderBlock(nn.Module):
    """
    Decoder Block: Upsample -> AttentionGate -> Concat -> ResidualConvBlock
    """
    def __init__(self, in_channels, skip_channels, out_channels, upsample_mode='convtranspose', norm=Norm.INSTANCE, act=Act.RELU, dropout=0.0, spatial_dims=3):
        super().__init__()
        inter_channels_att = skip_channels // 2 # Or some other factor

        if upsample_mode == 'convtranspose':
            self.upsample = nn.ConvTranspose3d(in_channels, in_channels, kernel_size=2, stride=2)
        elif upsample_mode == 'interpolate':
            self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        else:
            raise NotImplementedError(f"Upsample mode {upsample_mode} not implemented.")
             
        self.att_gate = AttentionGate(in_channels, skip_channels, inter_channels_att, norm=norm, act=act, spatial_dims=spatial_dims)
        self.res_block = ResidualConvBlock(in_channels + skip_channels, out_channels, stride=1, norm=norm, act=act, dropout=dropout, spatial_dims=spatial_dims)

    def forward(self, x, skip):
        #Upsample decoder signal
        x_up = self.upsample(x)
        #pad/crop x_up to match skip_att's spatial dimensions before attention gate
        if x_up.shape[2:] != skip.shape[2:]:
             # Simple padding/cropping - might need adjustment based on specific conv/padding choices
             diff_z = skip.shape[2] - x_up.shape[2]
             diff_y = skip.shape[3] - x_up.shape[3]
             diff_x = skip.shape[4] - x_up.shape[4]
             x_up = F.pad(x_up, [diff_x // 2, diff_x - diff_x // 2,
                                diff_y // 2, diff_y - diff_y // 2,
                                diff_z // 2, diff_z - diff_z // 2])
        skip_att = self.att_gate(x_up, skip)
        x_concat = torch.cat([x_up, skip_att], dim=1)
        out = self.res_block(x_concat)
        return out


class ResUNet(nn.Module):
    """
    3D Residual U-Net with Attention Gates.
    Inspired by: https://github.com/Akhilesh64/ResUnet-a/blob/main/model.py
    Adaptation to 3D and MONAI/PyTorch conventions.
    """
    def __init__(self, spatial_dims=3, in_channels=1, out_channels=1, channels=16, 
                 strides=2,
                 num_res_units=1, norm=Norm.INSTANCE, act=Act.RELU, dropout=0.1, upsample_mode='convtranspose'):
        super().__init__()
        
        if spatial_dims != 3:
            raise ValueError("This ResUNet implementation currently only supports spatial_dims=3.")
            
        if not isinstance(norm, str):
            norm = norm.value
        if not isinstance(act, str):
            act = act.value

        #list of filters based on the initial channel count
        initial_filters = channels
        f = [initial_filters * (2**i) for i in range(5)] #[16, 32, 64, 128, 256]
        
        pool_stride = strides if isinstance(strides, int) else strides[0] # Simple handling for now
        self.stem = ResidualConvBlock(in_channels, f[0], norm=norm, act=act, dropout=dropout, spatial_dims=spatial_dims)

        self.enc1 = EncoderBlock(f[0], f[1], pool_stride=pool_stride, norm=norm, act=act, dropout=dropout, spatial_dims=spatial_dims)
        self.enc2 = EncoderBlock(f[1], f[2], pool_stride=pool_stride, norm=norm, act=act, dropout=dropout, spatial_dims=spatial_dims)
        self.enc3 = EncoderBlock(f[2], f[3], pool_stride=pool_stride, norm=norm, act=act, dropout=dropout, spatial_dims=spatial_dims)

        self.bottleneck = ResidualConvBlock(f[3], f[4], norm=norm, act=act, dropout=dropout, spatial_dims=spatial_dims)

        self.dec1 = DecoderBlock(f[4], f[3], f[3], upsample_mode=upsample_mode, norm=norm, act=act, dropout=dropout, spatial_dims=spatial_dims)
        self.dec2 = DecoderBlock(f[3], f[2], f[2], upsample_mode=upsample_mode, norm=norm, act=act, dropout=dropout, spatial_dims=spatial_dims)
        self.dec3 = DecoderBlock(f[2], f[1], f[1], upsample_mode=upsample_mode, norm=norm, act=act, dropout=dropout, spatial_dims=spatial_dims)
        self.dec4 = DecoderBlock(f[1], f[0], f[0], upsample_mode=upsample_mode, norm=norm, act=act, dropout=dropout, spatial_dims=spatial_dims)
        self.final_conv = nn.Conv3d(f[0], out_channels, kernel_size=1)

    def forward(self, x):
        s0 = self.stem(x)
        s1, p1 = self.enc1(s0)
        s2, p2 = self.enc2(p1)
        s3, p3 = self.enc3(p2)

        b = self.bottleneck(p3)

        d1 = self.dec1(b, s3)
        d2 = self.dec2(d1, s2)
        d3 = self.dec3(d2, s1)
        d4 = self.dec4(d3, s0)

        output = self.final_conv(d4)
        return output

if __name__ == '__main__':
    # Example usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_channels = 1
    out_channels = 1 # 1 for binary segmentation
    initial_channels = 16
    norm_type = Norm.INSTANCE 
    act_type = Act.RELU
    dropout_rate = 0.1
    patch_size = (96, 96, 96) #example
    spatial_dims = len(patch_size)

    model = ResUNet(
        spatial_dims=spatial_dims, 
        in_channels=in_channels,
        out_channels=out_channels,
        channels=initial_channels, 
        strides=2,
        norm=norm_type,
        act=act_type,
        dropout=dropout_rate
    ).to(device)

    print(f"Model created on {device}")
    # Counting parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,}")
    # A simple test with a dummy input
    dummy_input = torch.randn(1, in_channels, *patch_size).to(device) # Batch size 1
    print(f"Input shape: {dummy_input.shape}")
    try:
        with torch.no_grad():
            output = model(dummy_input)
        print(f"Output shape: {output.shape}")
        # Check if output shape matches expected
        assert output.shape == (1, out_channels, *patch_size)
        print("Forward pass successful!")
        if device.type == 'cuda':
            print(f"Memory allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
            print(f"Memory reserved: {torch.cuda.memory_reserved(device) / 1024**2:.2f} MB")

    except Exception as e:
        print(f"Error during forward pass: {e}")

