import torch
from torch import nn
import torch.nn.functional as F
import torchvision

import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import pickle as pkl
from torch.utils.tensorboard import SummaryWriter

from .dataset import MultiH5Dataset
from .encoder import MAEEncoder
from .decoder import ConvDecoder

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

    
    fig, ax = plt.subplots(1,2, figsize=(12,6))
    ax[0].imshow(img_to_extract.permute(1,2,0))
    ax[1].matshow(csim)

    if verbose:
        plt.show()

    return t_nt, fig


def main():
    run_id = int(input('Enter run id: '))

    ds = MultiH5Dataset(list(DS_PATH.glob('train_100k/*'))) 
    print('Dataset size', len(ds))

    enc = MAEEncoder(d_model=384, patch_size=16).to(DEVICE)
    dec = ConvDecoder(d_model=384, patch_size=16).to(DEVICE)
    print('Enc', sum([p.numel() for p in enc.parameters()]))
    print('Dec', sum([p.numel() for p in dec.parameters()]))

    tb_writer = SummaryWriter(f'tensorboard_runs/m0_ae/run{run_id}')
    dl = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True)

    opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()), lr=0.001, weight_decay=0.05)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,len(dl),T_mult=2)
    crit = nn.MSELoss(reduction='none')

    for epoch in range(50):
        t = tqdm(dl, desc=f'Epoch {epoch+1}')
        t_nt = 0
        enc.train()
        dec.train()

        avg_loss = 0
        for z, x in enumerate(t):
            x = x.to(DEVICE)

            lat, mask = enc(x, mask_perc=0.75, return_mask=True)
            pred = dec(lat)

            loss = crit(x, pred)

            # Masked loss (predict only masked patches)
            loss_raw = crit(pred, x)
            loss = (loss_raw * mask.float().unsqueeze(1)).sum() / mask.sum() / 3

            loss.backward()
            opt.step()
            opt.zero_grad()

            t.set_postfix_str(f'Loss: {loss.item():.3f}')
            
            sch.step()

            avg_loss += loss.item()

        avg_loss /= len(dl)
        t_nt, scan_fig = score_model(enc, verbose=False)

        fig, ax = plt.subplots(1,3, figsize=(18,6))

        ax[0].imshow(x[0].cpu().permute(1,2,0))
        ax[1].imshow(pred[0].detach().cpu().permute(1,2,0))
        ax[2].imshow(mask[0].unsqueeze(0).detach().cpu().permute(1,2,0))

        tb_writer.add_scalar('Train loss', avg_loss, epoch)
        tb_writer.add_scalar('T_nt', t_nt, epoch)
        tb_writer.add_scalar('Learning rate', sch.get_last_lr()[0], epoch)

        tb_writer.add_figure('Scan fig', scan_fig, global_step=epoch,close=True)
        tb_writer.add_figure('Training fig', fig, global_step=epoch,close=True)

        if epoch%10==0:
            pkl.dump(enc, Path(f'src/m0_ae/run{run_id}_trained_model_e{epoch}.pkl').open('wb'))


    pkl.dump(enc, Path(f'src/m0_ae/run{run_id}_trained_model_e{epoch}.pkl').open('wb'))

if __name__=='__main__':
    main()