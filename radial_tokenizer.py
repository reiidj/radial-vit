import torch
import torch.nn as nn
import math

class RadialTokenizer(nn.Module):
    def __init__(self, feature_height, feature_width, in_channels, 
                 embed_dim, num_radial=4, num_angular=8, pooling='mean'):
        super().__init__()
        self.num_tokens = num_radial * num_angular
        self.pooling = pooling
        
        # Create polar coordinate grid
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, feature_height),
            torch.linspace(-1, 1, feature_width)
        )
        r = torch.sqrt(x**2 + y**2)
        theta = torch.atan2(y, x)
        
        # Create radial masks
        radial_bins = torch.linspace(0, 1, num_radial+1)[1:]
        angular_bins = torch.linspace(-math.pi, math.pi, num_angular+1)
        
        # Generate token masks
        masks = []
        for i in range(num_radial):
            for j in range(num_angular):
                radial_mask = (r >= radial_bins[i-1]) & (r < radial_bins[i]) if i > 0 else (r < radial_bins[0])
                angular_mask = (theta >= angular_bins[j]) & (theta < angular_bins[j+1])
                masks.append((radial_mask & angular_mask).float())
        
        self.register_buffer('masks', torch.stack(masks))
        self.proj = nn.Linear(in_channels, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_tokens, embed_dim))

    def forward(self, x):
        # x: (B, C, H, W)
        batch_size = x.shape[0]
        
        # Pool features using radial masks
        if self.pooling == 'mean':
            pooled = torch.einsum('bchw,thw->btc', x, self.masks) / self.masks.sum(dim=(1,2))
        elif self.pooling == 'max':
            pooled = x.unsqueeze(1) * self.masks.unsqueeze(0).unsqueeze(2)
            pooled, _ = pooled.flatten(3).max(dim=3)
        
        # Project and add positional embeddings
        tokens = self.proj(pooled) + self.pos_embed
        return tokens  # (B, num_tokens, embed_dim)