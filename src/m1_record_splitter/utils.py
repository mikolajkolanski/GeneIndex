import json
from pathlib import Path
from copy import deepcopy

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
import torch.nn.functional as F
from torchvision import tv_tensors

def read_labels(images_path:Path, labels_file_path:Path|None=None):
    if labels_file_path is None:
        labels_file_path = images_path / '0_labels.json'

    labs_json = json.load(labels_file_path.open())

    labs = []
    imgs = []   

    for k,v in labs_json.items():
        if len(v['regions']) == 0:
            continue
        imgs.append(images_path / v['filename'])
        l = []

        try:
            for rec in v['regions']:
                assert rec['shape_attributes']['name'] == 'rect'
                
                assert rec['shape_attributes']['x'] >0
                assert rec['shape_attributes']['y'] >0
                assert rec['shape_attributes']['width'] >0
                assert rec['shape_attributes']['height'] >0

                # assert rec['region_attributes'].get('act_num')
                l.append([rec['shape_attributes']['x'],
                        rec['shape_attributes']['y'],
                        rec['shape_attributes']['width'],
                        rec['shape_attributes']['height'],
                        int(rec['region_attributes'].get('act_num')) if rec['region_attributes'].get('act_num') else None])
        except Exception as e:
            raise Exception(k)
        labs.append(l)
    return labs,imgs


def show_bbox(img, targs,targs2=None,console=True,ret_fig=False):

    fig, ax = plt.subplots()
    ax.imshow(img.permute(1,2,0))
    
    for [x1,y1,x2,y2], lab in zip(targs['boxes'].tolist(),targs['labels'].tolist()):
        if lab==1:
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        else:
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='g', facecolor='none')
            ax.add_patch(rect)
    if targs2 is not None:
        for [x1,y1,x2,y2], lab in zip(targs2['boxes'].tolist(),targs2['labels'].tolist()):
            if lab==1:
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='b', facecolor='none')
                ax.add_patch(rect)
            else:
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='b', facecolor='none')
                ax.add_patch(rect)
    
    if console:
        plt.show()

    if ret_fig:
        return fig
        
def filter_rcnn_targets(targets, min_size=1.0):
    filtered_targets = []
    
    for target in targets:
        boxes = target.get('boxes')
        
        if boxes is None or len(boxes) == 0:
            filtered_targets.append(target)
            continue
            
        ws = boxes[:, 2] - boxes[:, 0]
        hs = boxes[:, 3] - boxes[:, 1]
        
        valid_mask = (ws > min_size) & (hs > min_size)
        
        if valid_mask.all():
            filtered_targets.append(target)
            continue
            
        filtered_target = {}
        num_boxes = len(boxes)
        
        for key, value in target.items():
            if isinstance(value, torch.Tensor) and len(value) == num_boxes:
                filtered_target[key] = value[valid_mask]
            else:
                filtered_target[key] = value
                
        filtered_targets.append(filtered_target)
        
    return filtered_targets

def feats_to_img(feats):
    Fdim, W, H = feats.shape
    feats_flat = feats.permute(1,2,0).reshape(-1, Fdim)
    csim = F.cosine_similarity(feats[:, 66:100, 20:80].mean(dim=(-1,-2)).unsqueeze(0), feats_flat, dim=1).reshape(1,W,H)

    return csim

def scale_targs(targs, factor, inplace=False):
    if not inplace:
        targs = deepcopy(targs)

    if not isinstance(targs, list):
        targs['boxes'] *= factor
        return targs
    else:
        for i in range(len(targs)):
            targs[i]['boxes'] *= factor
        return targs


def targs_to_bboxes(targs: list[dict], canvas_size):
    return [{'boxes': tv_tensors.BoundingBoxes(targs[i]['boxes'], format='XYXY', canvas_size=canvas_size),
            'labels': targs[i]['labels']}
            for i in range(len(targs))]
        

