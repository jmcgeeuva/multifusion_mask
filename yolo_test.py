import cv2
import pandas as pdb
# FIXME ME find a better way to address the ultralytics problem
cnt = 0
while cnt < 5:
    try:
        from ultralytics import YOLO
        cnt = 100
    except:
        cnt += 1
        pass
if cnt > 5 and cnt != 100:
    raise ValueError("Unable to load ultralytics try again")
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
import glob
from tqdm import tqdm
import threading
lock = threading.Lock()
plot_lock = threading.Lock()
# export PYTHONPATH=$PYTHONPATH:$(pwd)/IS-Fusion

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
    batch = [mask_tup[2] for entry in masks for mask_tup in entry]
    mask_entry = create_mask(mask_list, width, height)

    return mask_entry, labels, vids
    
def run_yolo8(model, tensor_list, width, height, device, search_labels= ['car', 'bicycle', 'fire hydrant', 'stop sign', 'parking meter', 'person']):
    """
    Given a tensor of BxAxCxHxW (batch x angle x channel x height x width) this function will run YOLO and find the segmentations for any of the search labels listed (as seen in model.names)
    these will then be output as a tensor of masks, the coinciding labels, and the number of the video that goes along with that output

    Note: If batch > 1 then angles and batches are combined and the coinciding masks can be found via the video_id
    """

    # if this input is BxAxCxHxW first rearrange to (BA)xCxHxW
    b, a, c, h, w = tensor_list.shape
    tensor_list = tensor_list.view(b*a, c, h, w)
    outputs = model(tensor_list, verbose=False)

    masks = [[(int(class_idx), mask, idx//a, idx%a) for mask, class_idx in zip(out.masks.data, out.boxes.cls) if model.names[int(class_idx)] in search_labels] 
             for idx, out in enumerate(outputs) if len(out.boxes.cls) > 0]

    labels =    [mask_tup[0] for entry in masks for mask_tup in entry]
    mask_list = [mask_tup[1] for entry in masks for mask_tup in entry]
    batches =   [mask_tup[2] for entry in masks for mask_tup in entry]
    angles =    [mask_tup[3] for entry in masks for mask_tup in entry]

    return mask_list, labels, batches, angles

def bbox_xyxy_from_mask_torch(mask: torch.Tensor):
    """
    mask: (B, H, W) tensor of 0/1 or bool
    returns: (B, 4) tensor of bounding boxes (x1, y1, x2, y2)
             if a mask has no positive pixels, it will return (0, 0, 0, 0) for that batch
    """
    assert mask.ndim == 3, "mask should be (B, H, W)"
    B, H, W = mask.shape
    mask = mask.bool()

    # Create coordinate grids
    y_coords = torch.arange(H, device=mask.device).view(1, H, 1)
    x_coords = torch.arange(W, device=mask.device).view(1, 1, W)

    # Masked positions (set non-object to inf/-inf for reduction)
    x_min = torch.where(mask, x_coords, torch.full_like(x_coords, W)).amin(dim=(1,2))
    x_max = torch.where(mask, x_coords, torch.full_like(x_coords, -1)).amax(dim=(1,2))
    y_min = torch.where(mask, y_coords, torch.full_like(y_coords, H)).amin(dim=(1,2))
    y_max = torch.where(mask, y_coords, torch.full_like(y_coords, -1)).amax(dim=(1,2))

    # Handle empty masks (where no 1s)
    empty = (x_max < 0) | (y_max < 0)
    x_min[empty] = 0
    y_min[empty] = 0
    x_max[empty] = 0
    y_max[empty] = 0

    # Stack into (B, 4)
    boxes = torch.stack([x_min, y_min, x_max, y_max], dim=1)
    return boxes

def process_camera_angle(model, tensor_list, mask_dir, orig_scale, search_labels= ['car', 'bicycle', 'fire hydrant', 'stop sign', 'parking meter', 'person']):
    """
    Given a tensor of BxAxCxHxW (batch x angle x channel x height x width) this function will run YOLO and find the segmentations for any of the search labels listed (as seen in model.names)
    these will then be output as a tensor of masks, the coinciding labels, and the number of the video that goes along with that output

    Note: If batch > 1 then angles and batches are combined and the coinciding masks can be found via the video_id
    """

    # if this input is BxAxCxHxW first rearrange to (BA)xCxHxW
    c, h, w = tensor_list.shape
    # tensor_list = tensor_list.view(c, h, w)
    with lock:
        out = model(tensor_list.unsqueeze(dim=0), verbose=False)[0]

    to_pil = transforms.ToPILImage()
    label_dir = {}
    save_dir = {}
    if len(out.boxes.cls) > 0:
        for mask, class_idx in zip(out.masks.data, out.boxes.cls):
            label = int(class_idx)
            label_name = model.names[label]

            bboxes = bbox_xyxy_from_mask_torch(mask.unsqueeze(dim=0)).squeeze(dim=0)
            areas = (bboxes[2] - bboxes[0])*(bboxes[3] - bboxes[1])
            total_area = mask.shape[-2]*mask.shape[-1]
            ratio = areas/total_area
            # Want the largest objects
            ratio_check = 0.002
            ratio_indices = (ratio > ratio_check).nonzero(as_tuple=True)[0]

            if label_name in search_labels and len(ratio_indices) > 0:
                if label_name not in label_dir.keys():
                    label_dir[label_name] = 0
                    save_dir[label_name] = []
                else:
                    label_dir[label_name] += 1

                save_dir[label_name].append(mask)

    os.makedirs(mask_dir, exist_ok=True)
    for label_name, pil_masks in save_dir.items():
        # get a mask and a label type
        save_dir = os.path.join(mask_dir, label_name)
        os.makedirs(save_dir, exist_ok=True)
        for itr, mask in enumerate(pil_masks):
            pil_mask = to_pil(mask * 255)
            pil_mask.save(os.path.join(save_dir, f'{itr}.jpg'))
                    
        # This is processing that exists to deal with when number of angles, detections, and 
        # batches don't match other batches and angles (FIXME maybe just fill in the rest with zeros?)
        # batches = idx//a
        # angles = idx%a
        # batches.append(batches)
        # angles.append(angles)

def plot_masks(mask_entry, labels, batches, angles, orig_images, separate=False, name='test'):
    if separate:
        cnt = 0
        for entry, label, batch, angle in zip(mask_entry, labels, batches, angles):
            plt.figure()
            plt.imshow(transforms.ToPILImage()(entry))
            plt.axis('off')
            plt.savefig(f'{name}{cnt}_{angle}.png')
            cnt += 1
    else:
        cnt = 0
        for entry, label, batch, angle in zip(mask_entry, labels, batches, angles):
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(transforms.ToPILImage()(entry))
            plt.axis('off')
            plt.subplot(1, 2, 2)
            plt.imshow(transforms.ToPILImage()(orig_images[batch, angle]))
            plt.axis('off')
            plt.savefig(f'{name}{cnt}.png')
            cnt += 1

def load_words(file_name):
    allowed_words = []
    with open(file_name) as f:
        for line in f:
            line = line.replace('\n', '')
            allowed_words.append(line)
    return allowed_words

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("angle", type=str, default=None)
    parser.add_argument("sample", type=str, default=None)
    parser.add_argument("--scale", type=int, default=32)
    parser.add_argument("--img_size", type=tuple, default=(1056, 384))
    parser.add_argument("--dir_root", type=str, default='./data/nuscenes/train')
    parser.add_argument("--allowed_words", type=str, default='./all_words.txt')
    parser.add_argument('--debug', action='store_true', help='Use debug if present')
    args = parser.parse_args()

    if args.angle is None or args.sample is None:
        print('ERROR: Invalid input need an angle')
        exit(0)

    model = YOLO("yolo26n-seg.pt")
    allowed_words = load_words(args.allowed_words)
    dir_root = args.dir_root
    angle = args.angle
    img_size = args.img_size
    sample = args.sample
    # FIXME turn into SLURM options
    # sample_locations = ['samples', 'sweeps']
    # [
    #     'CAM_FRONT',
    #     'CAM_BACK',
    #     'CAM_FRONT_LEFT',
    #     'CAM_FRONT_RIGHT',
    #     'CAM_BACK_LEFT',
    #     'CAM_BACK_RIGHT',
    # ]

    tqdm_lock = tqdm.get_lock()

    def task(model, files, mask_dir, allowed_words, progress_bar):
        for file in files:
            to_tensor = transforms.ToTensor()
            image = Image.open(file)
            orig_scale = (image.width, image.height)
            image_location = file.split('/')[-1].replace('.jpg', '')
            image_location = os.path.join(mask_dir, image_location)
            image = to_tensor(image.resize((1056, 384)))
            process_camera_angle(model, image, image_location, orig_scale, search_labels=allowed_words)
            with tqdm_lock:
                progress_bar.update(1)
    
    os.makedirs('./nuscenes_masks/train', exist_ok=True)
    mask_dir = os.path.join('./nuscenes_masks/train', sample, angle)
    os.makedirs(mask_dir, exist_ok=True)
    
    file_path = os.path.join(dir_root, sample, angle)
    files_in_data = glob.glob(f"{file_path}/*") 

    def split_list(lst, n_parts):
        k, m = divmod(len(lst), n_parts)
        return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n_parts)]

    num_threads = 10
    chunks = split_list(files_in_data, num_threads)

    import concurrent.futures as con
    with con.ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit tasks to the pool
        with tqdm(total=len(files_in_data)) as progress_bar:
            future_tasks = [executor.submit(task, model, files, mask_dir, allowed_words, progress_bar) for files in chunks]

            # THIS PART FIXES THE BAR
            for f in con.as_completed(future_tasks):
                f.result()  # propagate exceptions + ensure completion
            
        #     # Wait for all tasks to complete and retrieve results
        #     for future in con.as_completed(future_tasks):
        #         progress_bar.update(1)

    # for file in tqdm(files_in_data, total=len(files_in_data)): 
    #     task(model, file, mask_dir, allowed_words)
    
    # for thread in tqdm(threads, total=len(threads)):
    #     thread.join()

    # plot_masks(mask_entry, labels, vids)
    # if args.debug:
    #     # For printing
    #     masks= {idx:[(int(class_idx), mask) for mask, class_idx in zip(out.masks.xy, out.boxes.cls) if model.names[int(class_idx)] in search_labels] if len(out.boxes.cls) > 0 else [] for idx, out in enumerate(outputs)}
    #     fig, axes = plt.subplots(2, 3)
    #     row_cnt = 0
    #     col_cnt = 0
    #     for i, key in enumerate(masks.keys()):
    #         ax = axes[row_cnt, col_cnt]
    #         curr_image = orig_images[i]
    #         out_image = curr_image
    #         if len(masks[key]) > 0:
    #             out_image = draw_polygons(transforms.ToPILImage()(curr_image), [[mask[1] for mask in masks[key]]], [model.names[mask[0]] for mask in masks[key]], fill_mask=True, use_color=True)
    #             out_image = to_tensor(out_image)
    #         ax.axis('off')
    #         ax.set_title(f'CAM: {list(image_group.keys())[i]}')
    #         ax.imshow(out_image.permute(1, 2, 0))

    #         col_cnt += 1
    #         if col_cnt % 3 == 0:
    #             col_cnt = 0
    #             row_cnt += 1
    #     plt.savefig('test.png')

if __name__ == '__main__':
    main()