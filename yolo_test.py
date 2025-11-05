import cv2
import pandas as pdb
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# from tracker import *
import random
import numpy as np
import os
import argparse

from torchvision import transforms
import torch
import copy

def plot_bbox(image, bboxes, labels, name='test.png', print_label=True):
   # Create a figure and axes  
    fig, ax = plt.subplots()  
      
    # Display the image  
    ax.imshow(image)  
      
    # Plot each bounding box  
    for bbox, label in zip(bboxes, labels):  
        # Unpack the bounding box coordinates  
        x1, y1, w, h = bbox  
        # Create a Rectangle patch  
        rect = patches.Rectangle((x1-(w/2), y1-(h/2)), w, h, linewidth=1, edgecolor='r', facecolor='none')  
        # Add the rectangle to the Axes  
        ax.add_patch(rect)  
        # Annotate the label
        if print_label:  
            plt.text(x1, y1, label, color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))  
      
    # Remove the axis ticks and labels  
    ax.axis('off')  
      
    # Show the plot  
    plt.savefig(name) 

colormap = ['blue','orange','green','purple','brown','pink','gray','olive','cyan','red',
            'lime','indigo','violet','aqua','magenta','coral','gold','tan','skyblue', 'black']
def draw_polygons(image, polygons, labels, fill_mask=False, use_color=True, name='test.png', print_label=False):  
    """  
    Draws segmentation masks with polygons on an image.  
  
    Parameters:  
    - image_path: Path to the image file.  
    - prediction: Dictionary containing 'polygons' and 'labels' keys.  
                  'polygons' is a list of lists, each containing vertices of a polygon.  
                  'labels' is a list of labels corresponding to each polygon.  
    - fill_mask: Boolean indicating whether to fill the polygons with color.  
    """  
    # Load the image  
   
    output_image = copy.deepcopy(image)
    draw = ImageDraw.Draw(output_image)  
      
   
    # Set up scale factor if needed (use 1 if not scaling)  
    scale = 1  
      
    # Iterate over polygons and labels 
    for polygon, label in zip(polygons, labels):  
        # print(polygon, label) 
        if use_color:
            color = random.choice(colormap)  
            fill_color = random.choice(colormap) if fill_mask else None  
        else:
            color = 'black'
            fill_color = 'black'

        for _polygon in polygon: 
            _polygon = np.array(_polygon).reshape(-1, 2)  
            if len(_polygon) < 3:  
                print('Invalid polygon:', _polygon)  
                continue  
              
            _polygon = (_polygon * scale).reshape(-1).tolist()  
              
            # Draw the polygon  
            if fill_mask:  
                draw.polygon(_polygon, outline=color, fill=fill_color)  
            else:  
                draw.polygon(_polygon, outline=color)  
              
            # Draw the label text  
            if print_label:
                draw.text((_polygon[0] + 8, _polygon[1] + 2), label, fill=color)  
    
    return output_image

def create_mask(polygons, width, height, labels=None):  
    """  
    Draws segmentation masks with polygons on an image.  
  
    Parameters:  
    - image_path: Path to the image file.  
    - prediction: Dictionary containing 'polygons' and 'labels' keys.  
                  'polygons' is a list of lists, each containing vertices of a polygon.  
                  'labels' is a list of labels corresponding to each polygon.  
    - fill_mask: Boolean indicating whether to fill the polygons with color.  
    """  
    # Load the image  
   
      
   
    # Set up scale factor if needed (use 1 if not scaling)  
    scale = 1  
    masks = []
    transform = transforms.ToTensor()
    # Iterate over polygons and labels 
    # for idx, polygon in enumerate(polygons):  
    color = 'white'
    fill_color = 'white'

    for _polygon in polygons: 
        mask = Image.new("RGB", (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(mask)
        if labels is not None:
            label = labels[idx]
        _polygon = np.array(_polygon).reshape(-1, 2)  
        if len(_polygon) < 3:  
            print('Invalid polygon:', _polygon)  
            continue  
            
        _polygon = (_polygon * scale).reshape(-1).tolist()
            
        # Draw the polygon  
        # if fill_mask:  
        draw.polygon(_polygon, outline=color, fill=fill_color)  
        # else:  
        #     draw.polygon(_polygon, outline=color)  
            
        # Draw the label text  
        if labels is not None:
            draw.text((_polygon[0] + 8, _polygon[1] + 2), label, fill=color)  
        masks.append(transform(mask))
    return torch.stack(masks)
    
def run_yolo11(model, tensor_list, width, height, search_labels= ['car', 'bicycle', 'fire hydrant', 'stop sign', 'parking meter', 'person']):
    """
    Given a tensor of BxAxCxHxW (batch x angle x channel x height x width) this function will run YOLO and find the segmentations for any of the search labels listed (as seen in model.names)
    these will then be output as a tensor of masks, the coinciding labels, and the number of the video that goes along with that output

    Note: If batch > 1 then angles and batches are combined and the coinciding masks can be found via the video_id
    """

    # if this input is BxAxCxHxW first rearrange to (BA)xCxHxW
    b, a, c, h, w = tensor_list.shape
    tensor_list = tensor_list.view(b*a, c, h, w)
    outputs = model(tensor_list)

    masks = [[(int(class_idx), mask, video_id) for mask, class_idx in zip(out.masks.xy, out.boxes.cls) if model.names[int(class_idx)] in search_labels] 
             for video_id, out in enumerate(outputs) if len(out.boxes.cls) > 0]

    mask_list = [mask_tup[1] for entry in masks for mask_tup in entry]
    labels = [mask_tup[0] for entry in masks for mask_tup in entry]
    vids = [mask_tup[2] for entry in masks for mask_tup in entry]
    mask_entry = create_mask(mask_list, width, height)

    return mask_entry, labels, vids
    
def run_yolo8(model, tensor_list, width, height, search_labels= ['car', 'bicycle', 'fire hydrant', 'stop sign', 'parking meter', 'person']):
    """
    Given a tensor of BxAxCxHxW (batch x angle x channel x height x width) this function will run YOLO and find the segmentations for any of the search labels listed (as seen in model.names)
    these will then be output as a tensor of masks, the coinciding labels, and the number of the video that goes along with that output

    Note: If batch > 1 then angles and batches are combined and the coinciding masks can be found via the video_id
    """

    # if this input is BxAxCxHxW first rearrange to (BA)xCxHxW
    b, a, c, h, w = tensor_list.shape
    tensor_list = tensor_list.view(b*a, c, h, w)
    outputs = model(tensor_list)

    masks = [[(int(class_idx), mask, video_id) for mask, class_idx in zip(out.masks.data, out.boxes.cls) if model.names[int(class_idx)] in search_labels] 
             for video_id, out in enumerate(outputs) if len(out.boxes.cls) > 0]

    mask_list = [mask_tup[1] for entry in masks for mask_tup in entry]
    labels = [mask_tup[0] for entry in masks for mask_tup in entry]
    vids = [mask_tup[2] for entry in masks for mask_tup in entry]
    mask_entry = torch.stack(mask_list)

    return mask_entry, labels, vids

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=int, default=32)
    parser.add_argument('--debug', action='store_true', help='Use debug if present')
    args = parser.parse_args()

    model=YOLO('yolov8n-seg.pt')

    dir_root = './data/nuscenes/train/samples'
    scale = args.scale
    image_group = {
        "front_left":  'CAM_FRONT/n015-2018-08-01-16-32-59+0800__CAM_FRONT__1533112834612460.jpg',
        "front":       'CAM_BACK/n015-2018-08-01-16-32-59+0800__CAM_BACK__1533112834637525.jpg',
        "front_right": 'CAM_FRONT_LEFT/n015-2018-08-01-16-32-59+0800__CAM_FRONT_LEFT__1533112834604844.jpg',
        "back_left":   'CAM_FRONT_RIGHT/n015-2018-08-01-16-32-59+0800__CAM_FRONT_RIGHT__1533112834620339.jpg',
        "back":        'CAM_BACK_LEFT/n015-2018-08-01-16-32-59+0800__CAM_BACK_LEFT__1533112834647423.jpg',
        "back_right":  'CAM_BACK_RIGHT/n015-2018-08-01-16-32-59+0800__CAM_BACK_RIGHT__1533112834627893.jpg'
    }

    to_tensor = transforms.ToTensor()
    orig_images = []
    for angle, image_name in image_group.items():
        file_path = os.path.join(dir_root, image_name)
        image = Image.open(file_path)
        image = image.resize(((image.width//100)*scale, (image.height//100)*scale))
        orig_images.append(to_tensor(image))
    tensor_list = torch.stack(orig_images)

    mask_entry, labels, vids = run_yolo8(model, tensor_list.unsqueeze(dim=0), image.width, image.height)
    
    cnt = 0
    fig, axes = plt.subplots(2, mask_entry.shape[0])
    for entry, label, vid in zip(mask_entry, labels, vids):
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(transforms.ToPILImage()(entry))
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(transforms.ToPILImage()(orig_images[vid]))
        plt.axis('off')
        plt.savefig(f'test{cnt}.png')
        cnt += 1

    if args.debug:
        # For printing
        masks= {idx:[(int(class_idx), mask) for mask, class_idx in zip(out.masks.xy, out.boxes.cls) if model.names[int(class_idx)] in search_labels] if len(out.boxes.cls) > 0 else [] for idx, out in enumerate(outputs)}
        fig, axes = plt.subplots(2, 3)
        row_cnt = 0
        col_cnt = 0
        for i, key in enumerate(masks.keys()):
            ax = axes[row_cnt, col_cnt]
            curr_image = orig_images[i]
            out_image = curr_image
            if len(masks[key]) > 0:
                out_image = draw_polygons(transforms.ToPILImage()(curr_image), [[mask[1] for mask in masks[key]]], [model.names[mask[0]] for mask in masks[key]], fill_mask=True, use_color=True)
                out_image = to_tensor(out_image)
            ax.axis('off')
            ax.set_title(f'CAM: {list(image_group.keys())[i]}')
            ax.imshow(out_image.permute(1, 2, 0))

            col_cnt += 1
            if col_cnt % 3 == 0:
                col_cnt = 0
                row_cnt += 1
        plt.savefig('test.png')

if __name__ == '__main__':
    main()