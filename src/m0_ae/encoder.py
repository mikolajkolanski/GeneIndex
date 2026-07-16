import math

import torch
from torch import nn

from src.m0_ae.transformer import Block

SCAN_WIDTH = 1920
SCAN_HEIGHT = 1600

class MAEEncoder(nn.Module):
    def __init__(self, image_size=512, patch_size=16, d_model=256):
        super(MAEEncoder, self).__init__()
        
        self.num_patches = (image_size//patch_size)**2
        self.d_model = d_model
        self.img_size = image_size
        self.patch_size = patch_size

        self.patch_proj = nn.Conv2d(3, d_model, patch_size, patch_size)

        self.posenc = nn.Parameter(torch.randn(1, d_model, image_size//patch_size, image_size//patch_size))
        self.posenc_interp = nn.Parameter(torch.randn(1, d_model, SCAN_HEIGHT//patch_size, SCAN_WIDTH//patch_size))

        self.blocks = nn.ModuleList([
            Block(d_model, 8, 16),
            Block(d_model, 8, 16),
            Block(d_model, 8, 16),
            Block(d_model, 8, 0),

            Block(d_model, 8, 16),
            Block(d_model, 8, 16),
            Block(d_model, 8, 16),
            Block(d_model, 8, 0),
            
            Block(d_model, 8, 16),
            Block(d_model, 8, 16),
            Block(d_model, 8, 16),
            Block(d_model, 8, 0),
        ])

        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model))
    
    def update_posec_interp(self):
        with torch.no_grad():
            self.posenc_interp.copy_(torch.nn.functional.interpolate(self.posenc.detach(), 
                                                                    size=(SCAN_HEIGHT//self.patch_size, 
                                                                        SCAN_WIDTH//self.patch_size), 
                                                                    mode="bilinear"))


    def forward(self, x, mask_perc=0.7):
        if not self.training:
            mask_perc = 0

        p = self.patch_proj(x) # [N, d_model, x, y]

        pnum_x = p.size(-2)
        pnum_y = p.size(-1)

        seqlen = pnum_x * pnum_y
        
        if x.size(-1) == 512:
            p_emb = p + self.posenc
        else:
            assert x.size(-1) == 1920, f'scan must be of size 1920x1600, got {x.size()}'
            p_emb = p + self.posenc_interp
                                           
        p_emb = p_emb.flatten(-2).permute(0,2,1) # [N, 1024, d_model]


        rand = torch.rand(x.size(0), seqlen, device=x.device)
        shuffle_idx = rand.argsort(dim=1)
        idx_keep = shuffle_idx[:, 0:math.ceil((1-mask_perc)*seqlen)]
        idx_restore = shuffle_idx.argsort(dim=1)

        p_emb = torch.gather(
            p_emb,
            dim=1,
            index=idx_keep.unsqueeze(-1).expand(-1, -1, p_emb.size(-1))
        )

        for b in self.blocks:
            p_emb = b(p_emb, pnum_x, pnum_y) # [N, seqlen, d_model]
        latent = p_emb

        latent = torch.cat([latent, 
                            self.mask_token.repeat(x.size(0), seqlen-latent.size(1), 1)],
                            dim=1)

        lat_rest = torch.gather(
            latent,
            dim=1,
            index=idx_restore.unsqueeze(-1).expand(-1, -1, latent.size(-1))
        )
        lat_rest = lat_rest.permute(0, 2, 1).reshape(-1, self.d_model, pnum_x, pnum_y) # [N, 256, 32, 32]
        return lat_rest # , idx_keep.reshape()