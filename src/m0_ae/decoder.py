import torch
from torch import nn

class MAEDecoder(nn.Module):
    def __init__(self, image_size=512, patch_size=16, d_model=256):
        super(MAEDecoder, self).__init__()
        
        self.num_patches = (image_size//patch_size)**2
        self.d_model = d_model
        self.img_size = image_size
        self.patch_size = patch_size

        self.posenc = nn.Parameter(torch.randn(1, self.num_patches, d_model))

        self.blocks = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d_model, 8, 2*d_model,
            norm_first=True, 
            batch_first=True
        ), num_layers=2)

        # self.depatch = nn.ConvTranspose2d(d_model, 3, patch_size, patch_size)

        def dub(cin,cout,K,s,p):
            return nn.Sequential(
                nn.Conv2d(cin, cout, K, s, p), nn.BatchNorm2d(cout), nn.GELU(),
                nn.Conv2d(cout, cout, K, s, p), nn.BatchNorm2d(cout), nn.GELU(),
            )
        

        self.depatch = nn.Sequential( # Inp: [N, 256, 32, 32]
                nn.Upsample(scale_factor=2), 
                dub(self.d_model, 128,3,1,1), # [N, 128, 64, 64]
                nn.Upsample(scale_factor=2), 
                dub(128, 64,3,1,1), # [N, 64, 128, 128]
                nn.Upsample(scale_factor=2), 
                dub(64, 32,3,1,1), # [N, 32, 256, 256]
                nn.Upsample(scale_factor=2), 
                dub(32,16,3,1,1),
                # nn.Conv2d(32, 16,1,1,0), nn.BatchNorm2d(16), nn.LeakyReLU(),
                nn.Conv2d(16,3,1,1,0),
            )  

    def forward(self, x):
        p = x.flatten(-2).permute(0,2,1) # [N, 1024, 256]
        p_emb = p + self.posenc

        latent = self.blocks(p_emb) # [N, 1024, 256]
        latent = latent.permute(0, 2, 1).reshape(-1, self.d_model, self.img_size//self.patch_size, self.img_size//self.patch_size) 

        return self.depatch(latent)
    
class ConvDecoder(nn.Module):
    def __init__(self, image_size=512, patch_size=16, d_model=256):
        super(ConvDecoder, self).__init__()
        
        self.num_patches = (image_size//patch_size)**2
        self.d_model = d_model
        self.img_size = image_size
        self.patch_size = patch_size

        def dub(cin,cout,K,s,p):
            return nn.Sequential(
                nn.Conv2d(cin, cout, K, s, p), nn.BatchNorm2d(cout), nn.GELU(),
                nn.Conv2d(cout, cout, K, s, p), nn.BatchNorm2d(cout), nn.GELU(),
            )
        

        self.upscale = nn.Sequential( # Inp: [N, 256, 32, 32]
                nn.Upsample(scale_factor=2), 
                dub(self.d_model, 128,3,1,1), # [N, 128, 64, 64]
                nn.Upsample(scale_factor=2), 
                dub(128, 64,3,1,1), # [N, 64, 128, 128]
                nn.Upsample(scale_factor=2), 
                dub(64, 32,3,1,1), # [N, 32, 256, 256]
                nn.Upsample(scale_factor=2), 
                dub(32,16,3,1,1),
                # nn.Conv2d(32, 16,1,1,0), nn.BatchNorm2d(16), nn.LeakyReLU(),
                nn.Conv2d(16,3,1,1,0),
            )  

    def forward(self, x):
        return self.upscale(x)