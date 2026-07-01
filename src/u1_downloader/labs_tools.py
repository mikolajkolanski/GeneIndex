from functools import cache

import requests
import pandas as pd
from bs4 import BeautifulSoup

from src.types import *

def _filter_mask_easy(df: pd.DataFrame):
    return ((df.act_num != '') * (df.name != '') * (df.surname != '')*df.source_link.str.startswith('https://metryki.genealodzy.pl/'))


@cache
def _get_gid_from_ar_zsid(ar,zsid) -> GID:
    soup = BeautifulSoup(requests.get(f'https://skanoteka.genealodzy.pl/metryki.php?op=kt&ar={ar}&zs={zsid}').text)
    return soup.find('a', string="powrót do zespołu").get('href')[2::]


def get_img_id_from_labels(df):
    df_links = df[_filter_mask_easy(df)].source_link.str[len(' https://metryki.genealodzy.pl/')-1::].reset_index(drop=True)

    df_down = pd.DataFrame(columns=['gid','zsid', 'ar','sgid','vol'])
    df_down["gid"] = df_links.str.extract(r'(?:[?&]id=|\bid)([a-zA-Z0-9]+)')
    df_down["zsid"] = df_links.str.extract(r'(?:[?&]zs=|\bzs)([a-zA-Z0-9]+)')
    df_down["ar"] = df_links.str.extract(r'(?:[?&]ar=|\bar)([a-zA-Z0-9]+)')
    df_down["sgid"] = df_links.str.extract(r'(?:[?&]sy=|-sy)([a-zA-Z0-9]+)')
    df_down["vol"] = df_links.str.extract(r'(?:[?&]kt=|-kt)([a-zA-Z0-9]+)')

    df_down.loc[df_down.gid.isna(), 'gid'] = df_down.loc[df_down.gid.isna(), :].apply(lambda x: _get_gid_from_ar_zsid(x.ar,x.zsid),axis=1)
    df_down = df_down.drop(['zsid','ar'],axis=1)

    return df_down



