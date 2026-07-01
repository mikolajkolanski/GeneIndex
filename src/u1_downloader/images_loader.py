import os
import pandas as pd
from PIL import Image

def load_all(path) -> pd.DataFrame:
    image_paths = path +'/'+ pd.Series(os.listdir(path))
    print(f'Found {len(image_paths)} images')
    df = pd.DataFrame(list(image_paths),columns=['image_path'])
    df['image'] = df.image_path.map(Image.open)

    # def split_name(image_path):
    #     name = ''.join(image_path.split('/')[-1])[0:-4].split('-')
    #     return pd.Series([
    #         int(name[0]),
    #         int(name[1]),
    #         int(name[2]),
    #     ])
    # df[['gid', 'sgid', 'fileid']] = df.image_path.apply(split_name)
    df[["gid", "sgid", "vol", "fname"]] = df.image_path.str.extract(r'/gid([^-]*)-sgid([^-]*)-vol([^-]*)-name([^.]*).jpg')
    # df["zsid"] = df.image_path.str.extract(r'(?:[?&]zs=|\bzs)([a-zA-Z0-9]+)')

    def get_resolution(img):
        w,h = img.size
        return pd.Series([w,h])
    df[['width', 'height']] = df.image.apply(get_resolution)
    df['resolution'] = df.width.astype(str) + 'x' + df.height.astype(str)

    return df