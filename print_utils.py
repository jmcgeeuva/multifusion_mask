from mmcv import Config, DictAction
from mmdet.apis import set_random_seed
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
import torchvision.transforms as transforms
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from tqdm import tqdm
import mmcv
import torch
from mmdet3d.apis import collect_results_gpu, collect_results_cpu
from yolo_test import run_yolo8, plot_masks
import matplotlib.patches as patches
from mmcv.parallel import DataContainer as DC
import os
os.environ['YOLO_VERBOSE'] = 'False'
from ultralytics import YOLO
import argparse
import time
from copy import deepcopy
from torch import optim
from datetime import datetime
import numpy as np
from augmentation import get_augmentation
import random
# export PYTHONPATH=$PYTHONPATH:$(pwd)/IS-Fusion

def plot_bbox(image, bboxes, labels, print_labels=False, name='test.png'):
   # Create a figure and axes  
    fig, ax = plt.subplots()  
      
    # Display the image  
    ax.imshow(image.permute(1, 2, 0))  
      
    # Plot each bounding box  
    for bbox, label in zip(bboxes, labels):  
        # Unpack the bounding box coordinates  
        x1, y1, x2, y2 = bbox  
        # Create a Rectangle patch  
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')  
        # Add the rectangle to the Axes  
        ax.add_patch(rect)  
        # Annotate the label  
        if print_labels:
            plt.text(x1, y1, label, color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))  
      
    # Remove the axis ticks and labels  
    ax.axis('off')  
      
    # Show the plot  
    plt.savefig(f'{name}.png')  

def print_images(img, idx, separate=False, norm=True, name='test'):
    img = img[idx].permute(0, 2, 3, 1)

    if norm:
        img = (img - img.min()) / (img.max() - img.min())
        
    img1 = img[0, ...]
    img2 = img[1, ...]
    img3 = img[2, ...]
    img4 = img[3, ...]
    img5 = img[4, ...]
    img6 = img[5, ...]

    if separate:
        for cnt, (image, angle) in enumerate(zip([img1, img2, img3, img4, img5, img6], 
                                ["CAMERA_FRONT", "CAMERA_FRONT_RIGHT", "CAMERA_FRONT_LEFT", "CAMERA_BACK", "CAMERA_BACK_LEFT", "CAMERA_BACK_RIGHT"])):
            plt.figure()
            plt.imshow(image)
            plt.title(angle)
            plt.axis('off')
            plt.savefig(f'{name}_{cnt}_{angle}.png')
    else:
        plt.figure()
        fig, ax = plt.subplots(2, 3)

        ax[0, 0].imshow(img1)
        ax[0, 0].set_title(f"CAMERA_FRONT")
        ax[0, 0].axis('off')
        ax[0, 1].imshow(img2)
        ax[0, 1].set_title(f"CAMERA_FRONT_RIGHT")
        ax[0, 1].axis('off')
        ax[0, 2].imshow(img3)
        ax[0, 2].set_title(f"CAMERA_FRONT_LEFT")
        ax[0, 2].axis('off')
        ax[1, 0].imshow(img4)
        ax[1, 0].set_title(f"CAMERA_BACK")
        ax[1, 0].axis('off')
        ax[1, 1].imshow(img5)
        ax[1, 1].set_title(f"CAMERA_BACK_LEFT")
        ax[1, 1].axis('off')
        ax[1, 2].imshow(img6)
        ax[1, 2].set_title(f"CAMERA_BACK_RIGHT")
        ax[1, 2].axis('off')
        plt.tight_layout()
        plt.savefig(f'{name}.png')