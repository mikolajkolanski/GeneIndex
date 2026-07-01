import numpy as np
import cv2
from tqdm import tqdm

import matplotlib.pyplot as plt

def crop_and_split_pages(img,verbose=False):
    """Splits input image into two pages"""
    gray_image = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
    img_corners = cv2.adaptiveThreshold(
        gray_image, 
        255,                                  # Wartość dla tła (255 = idealna biel)
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,       # Metoda liczenia wagi (Gaussian jest płynniejszy)
        cv2.THRESH_BINARY,                    # Tryb: tekst na czarno, tło na biało
        3,                                   # Rozmiar bloku (musi być nieparzysty, np. 11, 21, 31)
        10                                    # Stała odejmowana od średniej
    )
    img_corners = cv2.GaussianBlur(img_corners, (5,5), 0)
    img_corners = 255 - img_corners

    if verbose:
        plt.imshow(img_corners)
        plt.show()

    hcut = (np.array(img_corners)).mean(axis=0)
    hcut_mean = np.mean(hcut)
    hcut_idxs = (hcut>hcut_mean).nonzero()[0]
    hcut1,hcut2 = hcut_idxs[0], hcut_idxs[-1]

    vcut = (np.array(img_corners)).mean(axis=1)
    vcut_mean = np.mean(vcut)
    vcut_idxs = (vcut>vcut_mean).nonzero()[0]
    vcut1,vcut2 = vcut_idxs[0], vcut_idxs[-1]

    hcut1,vcut1,hcut2,vcut2 = hcut1-30,vcut1-30,hcut2+30,vcut2+30 # Margins
    # vcut1 += 53 # Remove heading

    img = img.crop((hcut1,vcut1,hcut2,vcut2))
    page1 = img.crop((0,0,img.width//2,img.height))
    page2 = img.crop((img.width//2,0,img.width,img.height))
    return page1,page2


def get_h_conturs(img,verbose=False):
    """Extracts horizontal lines in the input image"""
    gray_image = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
    img_corners = cv2.adaptiveThreshold(
        gray_image, 
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        3,
        5
    )

    if verbose:
        plt.imshow(img_corners)
        plt.show()

    thresh = cv2.adaptiveThreshold(img_corners, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY_INV, 7, 5)

    kernel_length = np.array(img).shape[1] // 160
    # vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))

    # img_vw = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vert_kernel, iterations=2)
    img_hw = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, hori_kernel, iterations=2)

    img_hw = cv2.GaussianBlur(img_hw, (25,25), 13)
    if verbose:
        plt.imshow(img_hw)
        plt.show()
    return img_hw


def draw_lines(img,lines):
    """Draws lines from cv2.HoughLinesP for debug purposes"""
    plt.imshow(img)
    end_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line
            # if abs(np.rad2deg(np.arctan2(y1 - y2, x2 - x1))) < 4:
            plt.plot((x1, x2), (y1, y2),color='r')
            end_lines.append((x1,y1,x2,y2))
    print(end_lines)
    print(len(end_lines))
    plt.show()


def merge_lines(lines, dist_threshold=50) -> list:
    """Merges duplicated lines from cv2.HoughLinesP by averaging their positions"""
    if lines is None: return []

    processed_lines = []
    for line in lines:
        x1, y1, x2, y2 = line
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        processed_lines.append({'coords': (x1, y1, x2, y2), 'mid': (mid_x, mid_y), 'used': False})

    merged_lines = []

    for i in range(len(processed_lines)):
        if processed_lines[i]['used']: continue
        
        # Startujemy nową grupę
        current_group = [processed_lines[i]['coords']]
        processed_lines[i]['used'] = True
        
        for j in range(i + 1, len(processed_lines)):
            if processed_lines[j]['used']: continue

            # Sprawdź odległość między środkami linii
            dist = abs(processed_lines[i]['mid'][1] - processed_lines[j]['mid'][1])
            
            if dist < dist_threshold:
                current_group.append(processed_lines[j]['coords'])
                processed_lines[j]['used'] = True
        
        # Uśrednij współrzędne dla grupy (uproszczony model)
        avg_coords = np.mean(current_group, axis=0).astype(int)
        merged_lines.append(avg_coords)
        
    return merged_lines


def get_rows_pos(page,verbose=False) -> list[int]:
    """Returns y positions of rows on a page"""
    img_hw = get_h_conturs(page,verbose=verbose)

    min_length = 500
    lines = cv2.HoughLinesP(img_hw, 1, np.pi/180, threshold=250, 
                            minLineLength=min_length, maxLineGap=700)[:,0].copy() # .copy() needed for memory leaks

    lines = lines[abs(np.rad2deg(np.arctan2(lines[:,1] - lines[:,3], lines[:,2] - lines[:,0])))<4]

    if verbose:
        draw_lines(img_hw,lines)

    lines_m =merge_lines(lines,dist_threshold=100)
    if verbose:
        draw_lines(img_hw,lines_m)
    rows = np.array([int((y1+y2)/2) for (x1,y1,x2,y2) in lines_m])
    rows.sort()

    return rows


def split_into_rows(full_image,source_path,verbose=False) -> list[dict]:
    """Splits scan into list of rows data containing row image and other data"""
    rows_list = []

    page1,page2 = crop_and_split_pages(full_image)

    for page_side in [0,1]:
        p = page1 if page_side==0 else page2
        rows = get_rows_pos(p,verbose=verbose)

        margin=10
        for i in range(0,len(rows)-1):
            # row_img = p.crop((0,int(rows[i])-margin,p.width-1,int(rows[i+1])+margin)).copy()
            # row_img = row_img.crop((0,0,1738,400))
            # filename = temp_dir / f'{random.randint(100000000000,999999999999)}.jpg'
            # row_img.save(filename)
            # rows_imgs.append(Image.open(filename))
            # row = {'image': p.crop((0,int(rows[i])-margin,p.width-1,int(rows[i+1])+margin)).copy(),
            #                  'source': source_path, 'page_side': page_side, 'row_idx':i}

            rows_list.append([p.crop((0,int(rows[i])-margin,p.width-1,int(rows[i+1])+margin)).copy(),
                              source_path, page_side, i])
    return rows_list

def split_into_rows_batch(raw_df) -> list[dict]:
    """Batched split_into_rows()"""
    rows_list = []

    t = tqdm(range(len(raw_df)),desc='Spliting into rows')
    for i in t:
        sample = raw_df.loc[i]

        rows_list.extend(split_into_rows(sample.image,sample.image_path))

        t.set_postfix_str(f'Rows: {len(rows_list)}')
    
    return rows_list