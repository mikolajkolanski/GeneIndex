import requests
import logging
import json
from io import BytesIO
import math

from bs4 import BeautifulSoup
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

from src.types import *

DATASET_DIR = 'data/downloaded_images'
ALL_GIDS_PATH = 'data/all_gids.json'

logging.basicConfig(level=logging.INFO, filename="logs.log", filemode="a",
                    format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s')

# TODO: add 'kt' subgroups


def get_all_gids(force=False) -> list[GID]:
    if not force:
        try:
            return json.load(open(ALL_GIDS_PATH,'r'))
        except:
            pass

    gids = []

    for prov in ['DS', 'KP', 'LB', 'LS', 'LD', 'MP', 'MZ', 'OP', 'PK', 'PL', 'PM', 'SL', 'SK', 'WM', 'WP', 'ZP']:
        in_this_prov = 0

        r = requests.get(f'https://metryki.genealodzy.pl/woj-{prov}')
        soup = BeautifulSoup(r.text,features="html.parser")
        for e in soup.find_all('a', {'class':['po','wo']}):
            po = int(e['href'][5::])

            r = requests.get(f'https://metryki.genealodzy.pl/index.php?op=zs&standalone=true&pw={po}')
            soup2 = BeautifulSoup(r.text,features="html.parser")

            for e2 in soup2.find_all('a', {'class':'asc'}):
                gids.append(int(e2['href'][2::]))
                in_this_prov += 1
        print(prov,in_this_prov)
    print('Total:', len(gids))
    
    json.dump(gids, open(ALL_GIDS_PATH,'w'))

    return gids


def get_subgroups(gid: GID) -> list[str]:
    soup = BeautifulSoup(requests.get(f'https://skanoteka.genealodzy.pl/id{gid}').text,features="html.parser")
    
    subgroups = [] 
    for el in soup.find_all('a'):
        if (link:=el.get('href')).startswith(f'id{gid}-sy'):

            name = el.parent.parent.find_next('td').find_next('td').find_next('td').contents[0]
            # if name.startswith('U'):
            subgroups.append(link[len(f'id{gid}-sy')::])
    logging.debug(f'Found {len(subgroups)} subgroups in {gid}: {subgroups}')
    print(f'Found {len(subgroups)} subgroups in {gid}')
    return subgroups


def get_filenames(gid:GID,sgid:SGID,vol:VOL = 1) -> list[str]:
    """Returns filenames of all files in given volume"""
    r = requests.get(f'https://metryki.genealodzy.pl/id{gid}-sy{sgid}-kt{vol}')
    soup = BeautifulSoup(r.text,features="html.parser")

    out = []
    for e in soup.find_all('a', {'class': 'plik'}):
        filename = e.text.strip()

        # Skip covers
        if filename.startswith('z'): continue
        if 'Sk' in filename: continue
        if '000' in filename: continue

        out.append(filename)
    logging.debug(f'Found {len(out)} filenames in gid{gid} sgid{sgid} vol{vol}: {out}')
    return out


def _get_download_url(gid:GID,sgid:SGID,vol:VOL,filename):
    url = f'https://skanoteka.genealodzy.pl/index.php?op=pg&id={gid}&se=&sy={sgid}&kt={vol}&plik={filename}.jpg&x=0&y=0'
    r = requests.get(url)
    if r.status_code != 200:
        print('Status code',r.status_code)
    soup = BeautifulSoup(r.text,features="html.parser")
    return soup.find('img', {'title': 'Pobierz zdjęcie'}).parent.get('href')


def _download_img_from_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        logging.error(f'Error downloading image (status code {r.status_code}) at {url}')
        print('Status code',r.status_code) 
        return False
    return Image.open(BytesIO(r.content))


def _download_img_from_url_and_save(url,save_path):
    r = requests.get(url)
    if r.status_code != 200:
        logging.error(f'Error downloading image (status code {r.status_code}) at {url}')
        return False
    img = Image.open(BytesIO(r.content))
    try:
        img.save(save_path)
    except Exception as e:
        logging.error(f'Could not save image "{url}" at {save_path}: {e}')

# def _download_subgroup(gid:GID,sgid:SGID):
#     filenames = get_filenames(gid,sgid)

#     t = tqdm(filenames,desc=f'Downloading subgroup {sgid} from group {gid}')
#     for filename in t:
#         img_url = get_download_url(gid,sgid,1, filename)
#         while (img:=download_img(img_url)) == False:
#             print('Retrying')
#             pass

#         save_filename = DATASET_DIR + '/' + f'{gid}-{sgid}-{filename}.jpg'
#         img.save(save_filename)
#         t.set_postfix_str(f'Saved {save_filename}')


def download_volume(path, gid:GID,sgid:SGID,vol:VOL,percentage=1):
    """Downloads all images in given volume and saves in into 'path' directory"""

    filenames = get_filenames(gid,sgid,vol)
    if percentage != 1:
        filenames = pd.Series(filenames).sample(math.ceil(len(filenames)*percentage))

    with ThreadPoolExecutor(max_workers=30) as executor:
        futures = {executor.submit(_download_img_from_url_and_save, _get_download_url(gid,sgid,vol,filename), path + '/' + f'gid{gid}-sgid{sgid}-vol{vol}-name{filename}.jpg'): filename for filename in filenames}

        d = 0
        with tqdm(total=len(filenames), desc=f'Downloading images from gid{gid} sgid{sgid} vol{vol}') as t:
            for future in as_completed(futures):
                d += 1
                t.set_postfix_str(f'Downloaded {d} rows')
                t.update(1)
                


def download_group(path, gid:GID,percentage=1):   
    """Downloads all images in gid and saves in into 'path' directory"""
    subgroups = get_subgroups(gid)
    for sgid in subgroups:
        download_volume(path, gid,sgid,1,percentage) # TODO: Replace 1 with all avaliable volumes in pair (gid,sgid)


if __name__=='__main__':
    # download_subgroup(3856,3530)
    # gids = get_all_gids()
    pass