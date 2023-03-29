"""
get semantic segmentation annotations from coco data set.
"""
from PIL import Image
import imgviz
import argparse
import os
import tqdm
from pycocotools.coco import COCO
import shutil
import numpy as np
import cv2
 
def save_colored_mask(mask, save_path):
    lbl_pil = Image.fromarray(mask.astype(np.uint8), mode="L")
    colormap = imgviz.label_colormap()
    lbl_pil.putpalette(colormap.flatten())
    lbl_pil.save(save_path)
 
 
def main(args):
    annotation_file = os.path.join(args.input_dir, 'annotations.json')
    os.makedirs(os.path.join(args.input_dir, 'SegmentationClass'), exist_ok=True)
    #os.makedirs(os.path.join(args.input_dir, 'images'), exist_ok=True)
    coco = COCO(annotation_file)
    catIds = coco.getCatIds()
    imgIds = coco.getImgIds()
    print("catIds len:{}, imgIds len:{}".format(len(catIds), len(imgIds)))
    for imgId in tqdm.tqdm(imgIds, ncols=100):
        img = coco.loadImgs(imgId)[0]
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        if len(annIds) > 0:
            mask = coco.annToMask(anns[0]) * anns[0]['category_id']
            for i in range(len(anns) - 1):
                mask += coco.annToMask(anns[i + 1]) * anns[i + 1]['category_id']
	    
            #img_origin_path = os.path.join(args.input_dir, img['file_name'])
            #img_output_path = os.path.join(args.input_dir, 'images', img['file_name'])
            seg_output_path = os.path.join(args.input_dir, 'SegmentationClass', img['file_name'].replace('.jpg', '.png').split('\\')[1])
            #shutil.copy(img_origin_path, img_output_path)
            #save_colored_mask(mask, seg_output_path)
            cv2.imwrite(seg_output_path,mask)

 
 
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="./data_dataset_coco", type=str,
                        help="input dataset directory")
    #parser.add_argument("--split", default="train2017", type=str,
                        #help="train2017 or val2017")
    return parser.parse_args()
 
 
if __name__ == '__main__':
    args = get_args()
    main(args)
