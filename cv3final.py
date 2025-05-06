# Import Python Standard Library dependencies
import datetime
from glob import glob
import json
import math
import os
from pathlib import Path
import random
import cv2

# Import utility functions
from cjm_pytorch_utils.core import get_torch_device, set_seed, move_data_to_device
from cjm_pandas_utils.core import markdown_to_pandas
from cjm_pil_utils.core import resize_img
from cjm_torchvision_tfms.core import ResizeMax, PadSquare, CustomRandomIoUCrop

# Import matplotlib for creating plots
import matplotlib.pyplot as plt

# Import numpy
import numpy as np

# Import the pandas package
import pandas as pd

# Set options for Pandas DataFrame display
pd.set_option('max_colwidth', None)  # Do not truncate the contents of cells in the DataFrame
pd.set_option('display.max_rows', None)  # Display all rows in the DataFrame
pd.set_option('display.max_columns', None)  # Display all columns in the DataFrame

# Import PIL for image manipulation
from PIL import Image

# Import PyTorch dependencies
import torch
from torch.amp import autocast
import torch.nn as nn
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader
from torchtnt.utils import get_module_summary
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision.tv_tensors import BoundingBoxes, Mask
import torchvision.transforms.v2  as transforms
import utils

# Import Mask R-CNN
from torchvision.models.detection import MaskRCNN
from torchvision.models import ResNeXt101_64X4D_Weights, resnext101_64x4d
from torchvision.models.detection.backbone_utils import BackboneWithFPN

# Import tqdm for progress bar
from tqdm.auto import tqdm

# Import encoding mask functions
from utils_mask import encode_mask, read_maskfile

# Import mask utilities
from pycocotools import mask as mask_utils

# Import defaultdict
from collections import defaultdict

# Set random seed
seed = 1234
set_seed(seed)

# Set device to GPU if available
device = get_torch_device()
dtype = torch.float32
device, dtype

# Set root directory for dataset
ROOT = 'hw3-data-release'
train_dir = os.path.join(ROOT, "train")
test_dir = os.path.join(ROOT, "test_release")
train_dir = Path(train_dir)
test_dir = Path(test_dir)

# Define function to split dataset into training and validation sets
def split_train_val(
        train_dir, 
        val_ratio=0.2, 
        seed=42):
    all_folders = [f for f in os.listdir(train_dir) 
                   if os.path.isdir(os.path.join(train_dir, f))]
    random.seed(seed)
    random.shuffle(all_folders)
    val_count = int(len(all_folders) * val_ratio)
    val_folders = set(all_folders[:val_count])
    train_folders = set(all_folders[val_count:])
    return train_folders, val_folders

# Define function to process dataset and create COCO format annotations
def process_dataset(
        train_dir, 
        output_json, 
        folders_to_include):
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": f"class_{i}"} for i in range(1, 5)]
    }

    image_id = 1
    ann_id = 1
    
    for folder in folders_to_include:
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
                bbox = mask_utils.toBbox(rle).tolist()
                area = int(mask_utils.area(rle))
                iscrowd = 0

                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": class_id,
                    "segmentation": rle,
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": iscrowd
                })
                ann_id += 1

        image_id += 1

    with open(output_json, "w") as f:
        json.dump(coco, f)

# Creates training and validation jsons
# train_folder, val_folder = split_train_val(train_dir, val_ratio = 0.2, seed = 42)
# process_dataset(train_dir, "train_anno.json", train_folder)
# process_dataset(train_dir, "val_anno.json", val_folder)

# Loads the COCO dataset
class CellsDataset(Dataset):
    def __init__(
            self, 
            root, 
            annFile, 
            transforms=None):
        self.root = Path(root)
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms
        self.MAX_INSTANCES = 50

    def __getitem__(
            self, 
            index):
        img_id = self.ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = self.root / img_info['file_name']
        image = Image.open(img_path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        masks, boxes, labels, areas, iscrowd = [], [], [], [], []

        for ann in anns:
            mask = self.coco.annToMask(ann)
            masks.append(mask)
            labels.append(ann["category_id"])
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            areas.append(ann["area"])
            iscrowd.append(ann["iscrowd"])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
        image_id = torch.tensor([img_id])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.zeros((len(anns),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": image_id,
            "area": areas,
            "iscrowd": iscrowd
        }
        
        if target["masks"].shape[0] > self.MAX_INSTANCES:
                for key in ["boxes", "labels", "masks", "area", "iscrowd"]:
                    target[key] = target[key][:self.MAX_INSTANCES]
        if self.transforms:
            target["boxes"] = BoundingBoxes(target["boxes"], 
                                            format="XYXY", canvas_size=image.size[::-1])
            target["masks"] = Mask(target["masks"])

            sample = {
                "image": image,
                "boxes": target["boxes"],
                "labels": target["labels"],
                "masks": target["masks"]
            }

            sample = self.transforms(sample)

            image = sample["image"]
            target["boxes"] = sample["boxes"]
            target["labels"] = sample["labels"]
            target["masks"] = sample["masks"]

        return image, target
    
    def __len__(self):
        return len(self.ids)

# Function to get the model instance segmentation
def get_model_instance_segmentation(num_classes):
    resnext = resnext101_64x4d(weights=ResNeXt101_64X4D_Weights)
    body = nn.Sequential(
        resnext.conv1,
        resnext.bn1,
        resnext.relu,
        resnext.maxpool,
        resnext.layer1,
        resnext.layer2,
        resnext.layer3,
        resnext.layer4,
    )

    return_layers = {
        '4': '0',
        '5': '1',
        '6': '2',
        '7': '3',
    }

    backbone = BackboneWithFPN(
        body, 
        return_layers=return_layers, 
        in_channels_list=[256, 512, 1024, 2048],
        out_channels=128)

    model = MaskRCNN(backbone, num_classes=num_classes, trainable_backbone_layers=2)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    model.roi_heads.detections_per_img = 100
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
        in_features_mask, 
        hidden_layer, 
        num_classes)
    
    for param in model.backbone.parameters():
        param.requires_grad = True

    for param in model.parameters():
        param.requires_grad = True

    return model

# Initialize the model
model = get_model_instance_segmentation(num_classes=5)  # 4 classes + background
model.name = 'resnext101_64x4d_2'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model.to(device=device,dtype=dtype)

# Get summary of the model
test_inp = torch.randn(1, 3, 256, 256).to(device)

summary_df = markdown_to_pandas(f"{get_module_summary(model.eval(), [test_inp])}")

summary_df = summary_df[summary_df.index == 0]

summary_df.drop(['In size', 'Out size', 'Contains Uninitialized Parameters?'], axis=1)

# Class to define safe IoU Cropping
class SafeIoUCrop:
    def __init__(
            self, 
            crop_tfm, 
            min_h=64, 
            min_w=64, 
            max_ar=5.0):
        self.crop_tfm = crop_tfm
        self.min_h = min_h
        self.min_w = min_w
        self.max_ar = max_ar

    def __call__(
            self, 
            sample):
        img = sample["image"]
        h, w = img.size[-2:]
        ar = max(w / h, h / w)

        if h < self.min_h or w < self.min_w or ar > self.max_ar:
            print(f"Skipping IoUCrop for image {w}x{h}, AR={ar:.2f}")
            return sample

        return self.crop_tfm(sample)
# Define size of training images
train_sz = 512

# # Data augmentation
# Random IoU cropping
iou_crop = CustomRandomIoUCrop(min_scale=0.3, 
                               max_scale=1.0, 
                               min_aspect_ratio=0.5, 
                               max_aspect_ratio=5.0, 
                               sampler_options=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                               trials=10, 
                               jitter_factor=0.15)
safe_crop = SafeIoUCrop(iou_crop, max_ar=5.0)

# Create a `ResizeMax` object
resize_max = ResizeMax(max_sz=train_sz)

# Create a `PadSquare` object
pad_square = PadSquare(shift=True, fill=0)

# Compose transforms for data augmentation
data_aug_tfms = transforms.Compose(
    transforms=[
        safe_crop,
        transforms.ColorJitter(
                brightness = (0.875, 1.125),
                contrast = (0.5, 1.5),
                saturation = (0.5, 1.5),
                hue = (-0.05, 0.05),
        ),
        transforms.RandomGrayscale(),
        transforms.RandomEqualize(),
        transforms.RandomPosterize(bits=3, p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
    ],
)

# Compose transforms to resize and pad input images
resize_pad_tfm = transforms.Compose([
    resize_max, 
    pad_square,
    transforms.Resize([train_sz] * 2, antialias=True)
])

# Compose transforms to sanitize bounding boxes and normalize input data
final_tfms = transforms.Compose([
    transforms.ToImage(), 
    transforms.ToDtype(torch.float32, scale=True),
    transforms.SanitizeBoundingBoxes(),
])

# Define the transformations for training, testing and validation datasets
train_tfms = transforms.Compose([
    data_aug_tfms, 
    resize_pad_tfm, 
    final_tfms
])

final_tfms_test = transforms.Compose([
    transforms.ToImage(), 
    transforms.ToDtype(torch.float32, scale=True),
])
valid_tfms = transforms.Compose([resize_pad_tfm, final_tfms_test])

# Load training and validation datasets
dataset = CellsDataset("hw3-data-release/train", 
                       "train_anno.json", 
                       transforms=train_tfms)

dataset_val = CellsDataset("hw3-data-release/train", 
                           "val_anno.json", 
                           ransforms=valid_tfms)

# Pass datasets to DataLoader
data_loader = DataLoader(dataset, batch_size=4, 
                         shuffle=True, num_workers=0, 
                         collate_fn=utils.collate_fn)

data_loader_val = DataLoader(dataset_val, batch_size=4, 
                             shuffle=False, num_workers=0, 
                             collate_fn=utils.collate_fn)

# Define function to run single epoch
def run_epoch(model, dataloader, 
              optimizer, lr_scheduler, 
              device, scaler, 
              epoch_id, is_training):
    """
    Function to run a single training or evaluation epoch.
    
    Args:
        model: A PyTorch model to train or evaluate.
        dataloader: A PyTorch DataLoader providing the data.
        optimizer: The optimizer to use for training the model.
        loss_func: The loss function used for training.
        device: The device (CPU or GPU) to run the model on.
        scaler: Gradient scaler for mixed-precision training.
        is_training: Boolean flag indicating whether the model is in training or evaluation mode.
    
    Returns:
        The average loss for the epoch.
    """

    model.train()
    
    epoch_loss = 0
    progress_bar = tqdm(total=len(dataloader), desc="Train" if is_training else "Eval")
    
    for batch_id, (inputs, targets) in enumerate(dataloader):
        inputs = torch.stack(inputs).to(device)
        
        with autocast(torch.device(device).type):
            if is_training:
                losses = model(inputs.to(device), move_data_to_device(targets, device))
            else:
                with torch.no_grad():
                    losses = model(inputs.to(device), move_data_to_device(targets, device))
        
            loss = sum([loss for loss in losses.values()])

        if is_training:
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                old_scaler = scaler.get_scale()
                scaler.update()
                new_scaler = scaler.get_scale()
                if new_scaler >= old_scaler:
                    lr_scheduler.step()
            else:
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                
            optimizer.zero_grad()

        loss_item = loss.item()
        epoch_loss += loss_item

        progress_bar_dict = dict(loss=loss_item, avg_loss=epoch_loss/(batch_id + 1))
        if is_training:
            progress_bar_dict.update(lr=lr_scheduler.get_last_lr()[0])
        progress_bar.set_postfix(progress_bar_dict)
        progress_bar.update()

        if is_training:
            stop_training_message = f"Loss is NaN or infinite at epoch {epoch_id}, batch {batch_id}. Stopping training."
            assert not math.isnan(loss_item) and math.isfinite(loss_item), stop_training_message

    progress_bar.close()
    
    return epoch_loss / (batch_id +1)

# Define function to train model
def train_loop(model, 
               train_dataloader, 
               valid_dataloader, 
               optimizer,  
               lr_scheduler, 
               device, 
               epochs, 
               checkpoint_path, 
               use_scaler=False):
    """
    Main training loop.
    
    Args:
        model: A PyTorch model to train.
        train_dataloader: A PyTorch DataLoader providing the training data.
        valid_dataloader: A PyTorch DataLoader providing the validation data.
        optimizer: The optimizer to use for training the model.
        lr_scheduler: The learning rate scheduler.
        device: The device (CPU or GPU) to run the model on.
        epochs: The number of epochs to train for.
        checkpoint_path: The path where to save the best model checkpoint.
        use_scaler: Whether to scale graidents when using a CUDA device
    
    Returns:
        None
    """
    scaler = torch.amp.GradScaler() if device.type == 'cuda' and use_scaler else None
    best_loss = float('inf')

    loss_history = defaultdict(list)

    for epoch in tqdm(range(epochs), desc="Epochs"):

        torch.cuda.empty_cache()
        print(f"Starting epoch {epoch}")
        train_loss = run_epoch(model, train_dataloader, 
                               optimizer, lr_scheduler, 
                               device, scaler, 
                               epoch, is_training=True)

        with torch.no_grad():
            valid_loss = run_epoch(model, valid_dataloader, 
                                   None, None, 
                                   device, scaler, 
                                   epoch, is_training=False)
        loss_history["epoch"].append(epoch)
        loss_history["train_loss"].append(train_loss)
        loss_history["valid_loss"].append(valid_loss)
        loss_history["lr"].append(lr_scheduler.get_last_lr()[0])

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), checkpoint_path)

            training_metadata = {
                'best epoch': epoch,
                'best train_loss': train_loss,
                'best valid_loss': valid_loss, 
                'best learning_rate': lr_scheduler.get_last_lr()[0],
                'model_architecture': model.name
            }
            with open(Path(checkpoint_path.parent/'training_metadata.json'), 'w') as f:
                json.dump(training_metadata, f)
        with open(Path(checkpoint_path.parent/'loss_history.json'), 'w') as f:
            json.dump(loss_history, f)
        torch.cuda.empty_cache()

# Generate timestamp for the training session (Year-Month-Day_Hour_Minute_Second)
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Create a directory to store the checkpoints if it does not already exist
checkpoint_dir = Path(f"{timestamp}")

# Create the checkpoint directory if it does not already exist
checkpoint_dir.mkdir(parents=True, exist_ok=True)

# The model checkpoint path
checkpoint_path = checkpoint_dir/f"{model.name}.pth"

print(checkpoint_path)

# Learning rate for the model
lr = 5e-4

# Number of training epochs
epochs = 26

# AdamW optimizer; includes weight decay for regularization
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# Learning rate scheduler; adjusts the learning rate during training
lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                   max_lr=lr, 
                                                   total_steps=epochs*len(data_loader))

# Train the model
train_loop(model=model, 
           train_dataloader=data_loader,
           valid_dataloader=data_loader_val,
           optimizer=optimizer, 
           lr_scheduler=lr_scheduler, 
           device=torch.device(device), 
           epochs=epochs, 
           checkpoint_path=checkpoint_path,
           use_scaler=True)

# Load the model for inference
model.eval()
model.load_state_dict(torch.load(f"{checkpoint_path}"))

# Open the test dataset
with open("hw3-data-release/test_image_name_to_ids.json") as f:
    test_meta = json.load(f)

# Save the results of the test dataset to a JSON file
submission = []
for entry in test_meta:
    image_path = Path("hw3-data-release/test_release") / entry["file_name"]
    image = Image.open(image_path).convert("RGB")
    input_img = resize_img(image, target_sz=train_sz, divisor=1)
    min_img_scale = min(image.size) / min(input_img.size)
    image_tensor = transforms.Compose([transforms.ToImage(), 
                                       transforms.ToDtype(torch.float32, 
                                                          scale=True)])(input_img)[None].to(device)
    width, height = image.size

    with torch.no_grad():
        output = model(image_tensor)[0]

    for i in range(len(output["scores"])):
        if output["scores"][i] < 0.7:
            continue
        mask = output["masks"][i, 0].cpu().numpy() > 0.3
        resized_mask = cv2.resize(
            mask.astype(np.uint8),
            dsize=(width, height),
            interpolation=cv2.INTER_NEAREST,
            ).astype(np.uint8)
        rle = encode_mask(resized_mask)
        rle["size"] = [height, width]

        x1, y1, x2, y2 = output["boxes"][i].tolist()
        x1 *= min_img_scale
        y1 *= min_img_scale
        x2 *= min_img_scale
        y2 *= min_img_scale

        bbox = [x1, y1, x2 - x1, y2 - y1]

        submission.append({
            "image_id": entry["id"],
            "bbox": bbox,
            "score": float(output["scores"][i]),
            "category_id": int(output["labels"][i]),
            "segmentation": rle,
        })

with open("test-results.json", "w") as f:
    json.dump(submission, f)
    print("Submission file saved as test-results.json")

# Visualize the training and validation loss and learning rate
with open("2025-05-06_11-54-09\loss_history.json") as f:
    data = json.load(f)

plt.plot(data["epoch"], data["train_loss"],'go-', data["valid_loss"], 'bo-')
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss vs Epoch")
plt.legend(["train_loss", "valid_loss"])
plt.xlim(-1, len(data["epoch"]))
plt.ylim(0, 3.5)
plt.show()

plt.plot(data["epoch"], data["lr"], 'ro-')
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title("Learning Rate vs Epoch")
plt.legend(["learning_rate"])
plt.show()


