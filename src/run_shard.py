import os
import pandas as pd
import numpy as np
from PIL import Image
import random
from tqdm import tqdm

import u2_image_record_splitter_alg as u2_image_record_splitter_alg
from u2_image_record_splitter_alg.images_fext_dino import DinoFeatureExtractor
from u1_downloader import images_loader

feature_ext = DinoFeatureExtractor()

def do_shard(shard_id:int, df_raw,save_imgs=True):
    """Does full one shard pipeline"""
    rows_list = u2_image_record_splitter_alg.split_into_rows_batch(df_raw)
    rows_df = pd.DataFrame(rows_list, columns=['image','source','page_side','row_idx'])

    height_outliers = np.array([img.height>400 for img in rows_df.image])
    rows_df = rows_df[height_outliers==False].reset_index(drop=True)

    rows_df['emb'] = list(feature_ext.batch_extract(rows_df.image))

    if not save_imgs:
        rows_df = rows_df.drop(columns=['image'])
        rows_df.to_pickle(f'data/sharddumps/sharddump_{shard_id}.pkl')
        return

if __name__ == '__main__':
    df = images_loader.load_all('data/downloaded_images')

    print('Unique resolutions:', df.resolution.unique())
    df.resolution.value_counts().head(5)

    # Choose only one resolution
    df = df[df.resolution.eq('4100x3148')]
    df = df.reset_index(drop=True)
    print(f'{len(df)} images have correct resolution')

    b_size = 32
    for i in tqdm(range(len(df)//b_size),desc='Doing all'):
        do_shard(i*b_size, df[i*b_size:(i+1)*b_size].reset_index(),save_imgs=False)
        break