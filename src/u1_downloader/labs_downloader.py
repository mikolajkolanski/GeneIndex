import logging
import time
import threading
import math

import requests
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from stem import Signal
from stem.control import Controller
from stem import SocketError
from requests.exceptions import ReadTimeout

from src.types import *


logging.basicConfig(level=logging.INFO, filename="logs.log", filemode="a",
                    format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s')

ip_change_lock = threading.Lock()
last_ip_change_time = 0

MAX_RETRIES = 3
TIMEOUT = 20
MAX_WORKERS = 50

proxies = {
    'http': 'socks5://127.0.0.1:9150',
    'https': 'socks5://127.0.0.1:9150'
}


def get_current_ip():
    r = requests.get('https://api.ipify.org?format=json', proxies=proxies, timeout=10)
    return r.json()['ip']


def renew_tor_ip(control_port=9151, password=""):
    global last_ip_change_time
    try:
        with Controller.from_port(port=control_port) as controller:
            controller.authenticate(password=password)
            controller.signal(Signal.NEWNYM)

            time.sleep(5)
            last_ip_change_time = time.time()
    except SocketError:
        logging.error('Socket error, ensure tor browser is running')
        raise SocketError('Ensure tor browser is running')

def fetch_page(rid: RID,page,pred_num_rows=None):
    url = f'http://geneteka.genealodzy.pl/api/getAct.php?start={page*50}&length=50&rid={rid}'
    headers = {
        'Referer': 'http://www.geneteka.genealodzy.pl/index.php',
        'X-Requested-With': 'XMLHttpRequest',
    }
    for _ in range(MAX_RETRIES):
        try:
            r = requests.get(url, headers=headers, proxies=proxies, timeout=TIMEOUT) 
            if pred_num_rows is None: return page, r.json()

            r_count = int(r.json()['recordsFiltered'])

            if r_count == pred_num_rows:
                return page, r.json()

            with ip_change_lock:
                if time.time() - last_ip_change_time > 10:
                    logging.info(f"[Thread {page}] - Rate limited ({pred_num_rows}pred vs {r_count}seen). Changing IP...")
                    renew_tor_ip()

        except ReadTimeout as e:
            logging.error(f'[Thread {page}] - Read timeout ({TIMEOUT}) exceeded: {e}')
        except Exception as e:
            logging.error(f'[Thread {page}] - Error: {e}')

        time.sleep(10)
    
    logging.error(f'[Thread {page}] - Max retries ({MAX_RETRIES}) exceded')
    return page, None


def get_data_from_rid(rid: RID):
    renew_tor_ip() # Ensure correct number of rows
    time.sleep(5)
    pred_num_rows = int(fetch_page(rid,0)[1]['recordsFiltered'])
    assert pred_num_rows == int(fetch_page(rid,0)[1]['recordsFiltered'])

    max_pages = math.ceil(pred_num_rows/50)

    data_combined = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fetch_page, rid, page,pred_num_rows): page for page in range(max_pages)}
        
        with tqdm(total=max_pages, desc=f"Downloading labels from rid{rid}") as t:
            for future in as_completed(futures):
                page, data = future.result()

                if data is not None:
                    data_combined.extend(data['data'])
            
                t.set_postfix_str(f'Downloaded {len(data_combined)} rows')
                t.update(1)
    return data_combined


def get_data_batch(rids: list[RID]):
    data_combined = []
    for rid in rids:
        try:
            data_combined.extend(get_data_from_rid(rid))
        except Exception as e:
            print(f'Failed to dowload rid={rid}: {e}')
    return data_combined


def get_all_rid_in_province(province_name,df=True) -> pd.DataFrame:
    
    url = f'https://geneteka.genealodzy.pl/index.php?op=gt&w={province_name}'

    r = requests.get(url)
    if r.status_code != 200:
        print('error', r.status_code)
    soup = BeautifulSoup(r.text,features="html.parser")

    rid = []
    for parish in soup.find(id='sel_rid').find_all():
        
        if parish.get('value') == 'b': continue

        if (d:= parish.get('data-b')): rid.append([province_name, parish.text, 'birth', d])
        if (d:= parish.get('data-s')): rid.append([province_name, parish.text, 'mariage', d])
        if (d:= parish.get('data-d')): rid.append([province_name, parish.text, 'death', d])
    print(f'Rids for {province_name}: {len(rid)}')
    if df:
        df = pd.DataFrame(rid,columns=['province_name', 'loc_name','type','rid'])
        return df
    return rid


def get_all_rid(df=True,extract_extras=True):
    provinces = ['01ds', '02kp', '03lb','04ls','05ld','06mp','07mz','71wa','08op','09pk','10pl','11pm','12sl','13sk','14wm','15wp','16zp'] # Only Poland

    rid = []
    for p in provinces:
        rid.extend(get_all_rid_in_province(p,df=False))
        try:
            pass
        except Exception as e:
            print(f'Failed to dowload prov={p}: {e}')

    if df:
        df = pd.DataFrame(rid,columns=['province_name', 'loc_name','type','rid'])
        return df
    return rid


def extract_from_extras(extras: pd.Series):
    """Extracts extra info. ~50s per 100k"""

    out = []

    for text in tqdm(extras,desc='Extracting additional data from extras'):
        
        soup = BeautifulSoup(text,features="html.parser")
        i = ''
        z = ''
        a = ''
        source_link = ''
        for img in soup.find_all('img'):
            # print(img['src'],img['title'])
            if img['src'] == 'images/i.png':
                i = img['title'][len('Uwagi: ')::]
            elif img['src'] == 'images/z.png':
                z = img['title'][len('Miejsce przechowywania ksiąg: \r')::]
            elif img['src'] == 'images/a.png':
                a = img['title'][len('Indeks dodał: ')::]
            elif img['src'] == 'images/s.png':
                source_link = img.parent['href']
        out.append(pd.Series([i,z,a,source_link]))
    return out
    

if __name__ == '__main__':
    all_rid = get_all_rid()

    for rid in all_rid[all_rid.type=='birth'].sample(20).rid:
        df_data = get_data_from_rid(rid)

        df = pd.DataFrame(df_data,columns=['born_year','act_num','name','surname','father_name','mother_name','mother_surname','parish','location','extras'])
        df[['extras_i','extras_z','extras_a','source_link']] = extract_from_extras(df.extras)

        df.to_pickle(f'data/downloaded_labs/labels-{rid}.pkl')

        time.sleep(5)
