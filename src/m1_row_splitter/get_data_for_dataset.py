import pandas as pd

from src.u1_downloader.labs_downloader import get_all_rid, get_data_from_rid, extract_from_extras
from src.u1_downloader.images_downloader import download_volume
from src.u1_downloader.labs_tools import get_img_id_from_labels

TRAIN_DS_PATH = 'datasets/ds2_raw'

# NEVER RUN THIS

if __name__ == '__main__':
    all_rid = get_all_rid()

    data = []
    for rid in all_rid[all_rid.type=='birth'].sample(5).rid:
        data.extend(get_data_from_rid(rid))

    df_labs = pd.DataFrame(data,columns=['born_year','act_num','name','surname','father_name','mother_name','mother_surname','parish','location','extras'])
    df_labs[['extras_i','extras_z','extras_a','source_link']] = extract_from_extras(df_labs.extras)

    df_down = get_img_id_from_labels(df_labs)
    df_down = df_down.drop_duplicates()

    print(f'Dowloading {len(df_down)} volumes')
    for row in df_down.iloc:
        download_volume(TRAIN_DS_PATH, row.gid, row.sgid, row.vol,percentage=0.001)


    # df = images_loader.load_all('data/downloaded_images')

    # print('Unique resolutions:', df.resolution.unique())
    # df.resolution.value_counts().head(5) 

    # # Choose only one resolution
    # df = df[df.resolution.eq('4100x3148')]
    # df = df.reset_index(drop=True)
    # print(f'{len(df)} images have correct resolution')

    # for sample in df.iterrows():
    #     page1,page2 = u2_image_row_splitter.crop_and_split_pages(sample.image) 