import torch
from torch import nn
import torch.nn.functional as F
import torchvision

import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import pickle as pkl

from src.m0_ae.dataset import MultiH5Dataset
from src.m0_ae.encoder import MAEEncoder
from src.m0_ae.decoder import ConvDecoder

DS_PATH = Path('src/m0_ae/dataset/')
SCAN_WIDTH = 1920
SCAN_HEIGHT = 1600
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def score_model(encoder, verbose=False):
    img_to_extract = Image.open('src/m0_ae/dataset/scoring_ds/scoring2.jpg')
    img_to_extract = img_to_extract.resize((SCAN_WIDTH, SCAN_HEIGHT))

    img_to_extract = torchvision.transforms.functional.to_tensor(img_to_extract)
    encoder.eval()
    encoder.update_posec_interp()
    with torch.no_grad():
        feats = encoder(img_to_extract.unsqueeze(0).to(DEVICE)).squeeze(0).detach().cpu()
    encoder.train()

    Fdim, W, H = feats.shape
    feats_flat = feats.permute(1,2,0).reshape(-1, Fdim)


    # Text vs not text
    t_nt = 0
    t_nt += F.cosine_similarity(feats[:, 66:100, 20:80].mean(dim=(-1,-2)), feats[:, 20:40, 20:80].mean(dim=(-1,-2)), dim=0)

    csim = F.cosine_similarity(feats[:, 66:100, 20:80].mean(dim=(-1,-2)).unsqueeze(0), feats_flat, dim=1).reshape(W,H)

    if verbose:
        fig, ax = plt.subplots(1,2, figsize=(15,15))
        ax[0].imshow(img_to_extract.permute(1,2,0))
        ax[1].matshow(csim)
        plt.show()
    print(f'T_nt = {t_nt.item()}')

    return t_nt

ds = MultiH5Dataset(list(DS_PATH.glob('train_100k/*'))) 
print('Dataset size', len(ds))

enc = MAEEncoder(d_model=384, patch_size=16).to(DEVICE)
dec = ConvDecoder(d_model=384, patch_size=16).to(DEVICE)
print('Enc', sum([p.numel() for p in enc.parameters()]))
print('Dec', sum([p.numel() for p in dec.parameters()]))

dl = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True)

opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()), lr=0.001, weight_decay=0.05)
crit = nn.MSELoss()

for epoch in range(50):
    t = tqdm(dl, desc=f'Epoch {epoch+1}')
    t_nt = 0
    enc.train()
    dec.train()
    for z, x in enumerate(t):
        x = x.to(DEVICE)

        lat = enc(x, mask_perc=0.75)
        pred = dec(lat)

        loss = crit(x, pred)

        loss.backward()
        opt.step()
        opt.zero_grad()

        t.set_postfix_str(f'Loss: {loss.item():.3f} TvsNT: {t_nt}')

        if z%100 == 0:
            t_nt = score_model(enc, verbose=False)

pkl.dump(enc, Path(f'src/m0_ae/{input('Model filename: ')}.pkl').open('wb'))