**DOS Dataset**

DOS Dataset is a novel indoor deformable object segmentation dataset for sweeping robots, introduced in the paper "DOS Dataset: A Novel Indoor Deformable Object Segmentation Dataset for Sweeping Robots".

**Download**

The dataset can be downloaded from:

Google Drive

**Installation**

To use the dataset, you need to install [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) and [SegNeXt](https://github.com/Visual-Attention-Network/SegNeXt). Please refer to the corresponding repository for installation instructions.
The configuration files for the models mentioned in the paper can be found in the dos_config directory.

**Train**

Before training the models, you need to convert the raw annotation files in train_annotations/ and val_annotations/ to COCO or VOC format. This project uses COCO format data for model training. You can use the following scripts for conversion:

COCO format: 

'''python
python labelme2coco.py ${ANNOTATION_FILENAME} ${OUTPUT_FILENAME} --labels ${CATEGORY_FILE}
'''

VOC format: 

'python labelme2voc.py ${ANNOTATION_FILENAME} ${OUTPUT_FILENAME} --labels ${CATEGORY_FILE}'

The script cocotomask.py converts the COCO dataset to mask images with pixel values in the range of '[0,num_class-1]':
'python cocotomask.py ${COCO_DATASET_FILENAME}'

**Statistics**

The script dataset_analysis1.py counts the number of images containing a specific number of instances:

'python dataset_analysis1.py ${INPUT_ANNOTATION_DIRECTORY}'

The script dataset_analysis2.py provides statistics on image resolution, number of instances, number of instances for each category, and annotation area for each instance:

'python dataset_analysis2.py ${COCO_ANNOTATION_JSON_FILENAME}'

**Inference**

mscan_l.pth is the pre-trained model for SegNeXt, and best_mIoU_iter_160000.pth is the model trained on DOS Dataset using SegNeXt. They can be downloaded from:

Google Drive

You can use the example scripts video_demo.py and image_demo.py in the demo directory to perform segmentation on videos or images using SegNeXt. We also provide a sample video contact_4.mp4.

**Notes**

For the first submission, only submit the files that need to be added to the [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) and [SegNeXt](https://github.com/Visual-Attention-Network/SegNeXt).
