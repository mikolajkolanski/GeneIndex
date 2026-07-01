import logging
import time
from functools import partial

import requests
from bs4 import BeautifulSoup


logging.basicConfig(level=logging.INFO, filename="logs.log", filemode="a",
                    format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s')

MAXRETRIES = 3

def get(url,delay=3,bs=False) -> BeautifulSoup | any:
    for att in range(MAXRETRIES):
        if att!=0:
            time.sleep(delay)
        logging.debug(f'Get (attempt {att}): {url}')
        try:
            r = requests.get(url)
        except Exception as e:
            logging.warning(f'[{url}] Request: {e}')
            continue

        if r.status_code != 200:
            logging.warning(f'[{url}] Status code {r.status_code}: {r.text}')
            continue
            
        x = r.content
        if bs:
            try:
                x = BeautifulSoup(r.text,features='html.parser')
            except Exception as e: 
                logging.warning(f'[{url}] Could not parse html: {e}')
                continue
            
        return x 
    logging.error(f'[{url}] Max retries ({MAXRETRIES}) exceeded.')
    raise ConnectionError(f'[{url}] Max retries ({MAXRETRIES}) exceeded.')
     

get_bs = partial(get, bs=True)

def proxy_get(url):
    pass