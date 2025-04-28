import os
from pathlib import Path
import cv2
import skimage.io as sio
import matplotlib.pyplot as plt
import numpy as np
# import tqdm as tqdm
import json
from glob import glob
import sys
import random
import math
import re
import time
import matplotlib

from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

from utils import encode_mask, decode_maskobj, read_maskfile

ROOT = 'hw3-data-release'
train_dir = os.path.join(ROOT, "train")
test_dir = os.path.join(ROOT, "test_release")
train_dir = Path(train_dir)
test_dir = Path(test_dir)

num_classes = 4

def process_dataset(train_dir, output_json):
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": f"class_{i}"} for i in range(num_classes)]
    }

    image_id = 1
    ann_id = 1
    for folder in os.listdir(train_dir):
        folder_path = os.path.join(train_dir, folder)
        if not os.path.isdir(folder_path): continue
        image_path = os.path.join(folder_path, "image.tif")
        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]
        coco["images"].append({
            "id": image_id,
            "file_name": f"{folder}/image.tif",
            "height": height,
            "width": width,
        })
        
        for mask_path in glob(os.path.join(folder_path, "class*.tif")):
            class_id = int(os.path.basename(mask_path).replace("class", "").replace(".tif", ""))
            instance_ids = read_maskfile(mask_path)

            for inst_val in np.unique(instance_ids):
                if inst_val == 0:
                    continue
                binary_mask = (instance_ids == inst_val)
                rle = encode_mask(binary_mask)
                from pycocotools import mask as mask_utils
                bbox = mask_utils.toBbox(rle).tolist()
                area = int(mask_utils.area(rle))

                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": class_id,
                    "segmentation": rle,
                    "bbox": bbox,
                    "area": area,
                })
                ann_id += 1

        image_id += 1

    with open(output_json, "w") as f:
        json.dump(coco, f)
        
# process_dataset(train_dir, "train_anno.json")

class CellsDataset(utils.Dataset):
    def load_cells(self, dataset_dir, subset):
        """Load a subset of the dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Load the JSON file
        annotation_path = f"{subset}_anno.json"
        with open(annotation_path) as f:
            coco = json.load(f)

        # Add classes
        for category in coco["categories"]:
            self.add_class("cells", category["id"], category["name"])

        # Add images
        for img in coco["images"]:
            image_path = os.path.join(train_dir, img["file_name"])  # Adjust path here
            self.add_image(
                "cells",
                image_id=img["id"],
                path=image_path,
                width=img["width"],
                height=img["height"]
            )

        # Map image_id -> annotations
        self.image_id_to_annotations = {}
        for ann in coco["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.image_id_to_annotations:
                self.image_id_to_annotations[img_id] = []
            self.image_id_to_annotations[img_id].append(ann)

    def load_mask(self, image_id):
        """Load instance masks for the given image ID."""
        info = self.image_info[image_id]
        img_id = info["id"]

        annotations = self.image_id_to_annotations.get(img_id, [])
        masks = []
        class_ids = []

        for ann in annotations:
            rle = ann['segmentation']
            mask = decode_maskobj(rle)
            masks.append(mask)
            class_ids.append(ann['category_id'])

        if masks:
            masks = np.stack(masks, axis=-1)
            class_ids = np.array(class_ids, dtype=np.int32)
            return masks, class_ids
        else:
            # No masks
            masks = np.zeros((info["height"], info["width"], 0), dtype=np.bool_)
            class_ids = np.array([], dtype=np.int32)
            return masks, class_ids
        
dataset_train = CellsDataset()
dataset_train.load_cells(ROOT, "train")
dataset_train.prepare()

dataset_val = dataset_train

MODEL_DIR = os.path.join(ROOT, "logs")

COCO_MODEL_PATH = os.path.join(ROOT, "mask_rcnn_coco.h5")

class CellsConfig(Config):
    NAME = "cells"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 4  # background + 4 classes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 64

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5
    
config = CellsConfig()
# config.display()

def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last(), by_name=True)

model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE, 
            epochs=20, 
            layers='heads')

model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE / 10,
            epochs=30, 
            layers="all")

model_path = os.path.join(MODEL_DIR, "mask_rcnn_2.h5")
model.keras_model.save_weights(model_path)



class InferenceConfig(CellsConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(MODEL_DIR, "mask_rcnn_test.h5")
model_path = model.find_last()

# Load trained weights
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

# image_id = random.choice(dataset_val.image_ids)
# original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
#     modellib.load_image_gt(dataset_val, inference_config, 
#                            image_id)

# log("original_image", original_image)
# log("image_meta", image_meta)
# log("gt_class_id", gt_class_id)
# log("gt_bbox", gt_bbox)
# log("gt_mask", gt_mask)

# visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, 
#                             dataset_train.class_names, figsize=(8, 8))


# results = model.detect([original_image], verbose=1)

# r = results[0]
# visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], 
#                             dataset_val.class_names, r['scores'])

with open("hw3-data-release/test_image_name_to_ids.json") as f:
    test_meta = json.load(f)

submission = []

for entry in test_meta:
    image_path = test_dir / entry["file_name"]
    image = sio.imread(str(image_path))  # skimage.io.imread
    image = image[..., :3]
    molded_image = modellib.mold_image(image, inference_config)
    molded_image = np.expand_dims(molded_image, 0)

    # Run detection
    results = model.detect([image], verbose=0)
    r = results[0]

    # For each detection, add to submission
    for i in range(r['rois'].shape[0]):
        mask = r['masks'][:, :, i]
        rle = encode_mask(mask)
        decoded_mask = decode_maskobj(rle)
        y1, x1, y2, x2 = r['rois'][i]
        bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]

        submission.append({
            "image_id": entry["id"],       # ID from test_image_name_to_ids.json
            "bbox": bbox,
            "score": float(r['scores'][i]),
            "category_id": int(r['class_ids'][i]),
            "segmentation": rle,
        })

# 4. Save submission JSON
with open("test-results.json", "w") as f:
    json.dump(submission, f)



    image_ids = np.random.choice(dataset_val.image_ids, 10)
    APs = []
    for image_id in image_ids:
        # Load image and ground truth data    
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset_val, inference_config,
                                   image_id)
        molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
        # Run object detection
        results = model.detect([image], verbose=0)
        r = results[0]
        # Compute AP
        AP, precisions, recalls, overlaps =\
            utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                             r["rois"], r["class_ids"], r["scores"], r['masks'])
        APs.append(AP)
        
    print("mAP: ", np.mean(APs))
