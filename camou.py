# https://huggingface.co/microsoft/Florence-2-large/blob/main/sample_inference.ipynb
import torch
from torchvision import datasets
import torchvision.transforms as transforms
import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import florence_pytorch.florence.modeling_florence2 as flor2
from florence_pytorch.florence.configuration_florence2 import *
from florence_pytorch.florence.florence_attn import *
from florence_pytorch.florence.utils import unnormalize, normalize

import numpy as np

import matplotlib.pyplot as plt  
import matplotlib.patches as patches  
import math
import random

# from mmcv import Config, DictAction
# from mmdet.apis import set_random_seed
# from mmdet3d.datasets import build_dataloader, build_dataset
# import matplotlib.pyplot as plt
# from tqdm import tqdm

def run_task(flo_model, processor, image, task_prompt, device, text_prompt=''):
    task_prompt = f'<{task_prompt}>'
    if text_prompt != '':
        prompt = f'{task_prompt}{text_prompt}'
    else:
        prompt = task_prompt

    inputs = processor(text=prompt, images=image, return_tensors="pt")

    input_ids = inputs["input_ids"].to(device)
    pixel_values = inputs["pixel_values"].to(device)
    generated_ids = flo_model.generate(
        input_ids=input_ids,
        pixel_values=pixel_values,
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)

    # print(generated_text)

    parsed_answer = processor.post_process_generation(
        generated_text[0],
        task=task_prompt,
        image_size=(image.size[0], image.size[1])
    )
    return parsed_answer, task_prompt

def plot_bbox(image, bboxes, labels, name='test.png'):
   # Create a figure and axes  
    fig, ax = plt.subplots()  
      
    # Display the image  
    ax.imshow(image)  
      
    # Plot each bounding box  
    for bbox, label in zip(bboxes, labels):  
        # Unpack the bounding box coordinates  
        x1, y1, x2, y2 = bbox  
        # Create a Rectangle patch  
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')  
        # Add the rectangle to the Axes  
        ax.add_patch(rect)  
        # Annotate the label  
        plt.text(x1, y1, label, color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))  
      
    # Remove the axis ticks and labels  
    ax.axis('off')  
      
    # Show the plot  
    plt.savefig(name)  


colormap = ['blue','orange','green','purple','brown','pink','gray','olive','cyan','red',
            'lime','indigo','violet','aqua','magenta','coral','gold','tan','skyblue', 'black']
def draw_polygons(image, polygons, labels, fill_mask=False, use_color=True):  
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
   
    draw = ImageDraw.Draw(image)  
      
   
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
            draw.text((_polygon[0] + 8, _polygon[1] + 2), label, fill=color)  


def create_mask(image, polygons, labels, fill_mask=False, use_color=True):  
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
    for polygon, label in zip(polygons, labels):  
        mask = Image.new("RGB", (image.width, image.height), (0, 0, 0))
        draw = ImageDraw.Draw(mask)
        color = 'white'
        fill_color = 'white'

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
            # draw.text((_polygon[0] + 8, _polygon[1] + 2), label, fill=color)  
            masks.append(transform(mask))
    return torch.stack(masks)

import re
def convert_str_to_bb(s):
    result = [int(x) for x in re.findall(r"<loc_(\d+)>", s)]
    return result

def convert_bb_to_str(bb):
    out_str = ''
    for b in bb:
        out_str += f'<loc_{math.ceil(b)}>'
    return out_str

class Args:
    out=None
    eval='bbox'
    format_only=False
    show=False
    config='IS-Fusion/configs/isfusion/isfusion_0075voxel.py'
    cfg_options=None
    bs=1
    launcher='none'
    seed=0
    deterministic=False

def main():
    args=Args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
            'results / save the results) with the argument "--out", "--eval"'
            ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    cfg.data.test.update(dict(samples_per_gpu=args.bs))

    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
        cfg.data.workers_per_gpu = 1
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    # export PYTHONPATH=$PYTHONPATH:$(pwd)/IS-Fusion
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

def print_images(dataset, idx):
    img = dataset[idx]['img'][0].data
    plt.figure()
    fig, ax = plt.subplots(2, 3)
    images = img.permute(0, 2, 3, 1)

    img1 = images[0, ...]
    img1 = (img1 - img1.min()) / (img1.max() - img1.min())
    ax[0, 0].imshow(img1)
    ax[0, 0].set_title(f"CAMERA_FRONT")
    
    img2 = images[1, ...]
    img2 = (img2 - img2.min()) / (img2.max() - img2.min())
    ax[0, 1].imshow(img2)
    ax[0, 1].set_title(f"CAMERA_FRONT_RIGHT")
    
    img3 = images[2, ...]
    img3 = (img3 - img3.min()) / (img3.max() - img3.min())
    ax[0, 2].imshow(img3)
    ax[0, 2].set_title(f"CAMERA_FRONT_LEFT")
    
    img4 = images[3, ...]
    img4 = (img4 - img4.min()) / (img4.max() - img4.min())
    ax[1, 0].imshow(img4)
    ax[1, 0].set_title(f"CAMERA_BACK")
    
    img5 = images[4, ...]
    img5 = (img5 - img5.min()) / (img5.max() - img5.min())
    ax[1, 1].imshow(img5)
    ax[1, 1].set_title(f"CAMERA_BACK_LEFT")
    
    img6 = images[5, ...]
    img6 = (img6 - img6.min()) / (img6.max() - img6.min())
    ax[1, 2].imshow(img6)
    ax[1, 2].set_title(f"CAMERA_BACK_RIGHT")

    plt.tight_layout()
    plt.savefig('test.png')
    return img1, img2, img3, img4, img5, img6


# for _ in tqdm(range(2000), total=2000):
#     t = next(iter(data_loader))
#     # torch.Size([1, 6, 3, 384, 1056])
#     # image = t['img'][0].data[0][0, 1, ...]
# img1, img2, img3, img4, img5, img6 = print_images(dataset, 200)


# Add florence logic from camou.py
device = "cuda:0"

flo_model, processor = flor2.load("LARGE_FT", device, revision=None)

# Create a mask with an id that can be used to mask the image 

def unnormalize(bb, w, h):
    """unnormalize
    Changes the florence style bounding box into the picture's scale
    """
    new_bb = []
    for i, b in enumerate(bb):
        if i % 2 != 0:
            new_bb.append((b/1000)*h)
        else:
            new_bb.append((b/1000)*w)
    return new_bb

def normalize(bb, w, h):
    """normalize
    Changes the bounding box given into florence style bounding boxes
    """
    new_bb = []
    for i, b in enumerate(bb):
        if i % 2 != 0:
            new_bb.append(math.ceil((b/h)*1000))
        else:
            new_bb.append(math.ceil((b/w)*1000))
    return new_bb

def run_camou(image, size_ratio_threshold = 1e-2, debug=False):
    image_area = image.height*image.width
    # import pdb; pdb.set_trace()

    # 1. Get all object bounding boxes and their labels
    task = 'OD'
    task_output, task_label = run_task(flo_model, processor, image, task, device, '') 

    bbs = task_output[task_label]['bboxes']
    bb_labels = task_output[task_label]['labels']

    # 2. Filter out all the cars (and trucks if you want)
    labels = []
    filtered_polygons = []
    copy_image = copy.deepcopy(image)
    for bb, label in zip(bbs, bb_labels):
        print(label)
        if label == 'car' or 'vehicle' in label or label == 'person':
            bb_str = convert_bb_to_str(normalize(bb, image.width, image.height))
            output = run_task(flo_model, processor, copy_image, 'REGION_TO_SEGMENTATION', device, bb_str)
            task_output, task_label = output
            polygon = [task_output[task_label]['polygons'][0][0]]
            labels.append(label)
            
            width = bb[2] - bb[0]
            height = bb[3] - bb[1]
            area = width*height
            
            ratio = area/image_area
            # if ratio > size_ratio_threshold:
            filtered_polygons.append(polygon)

    masks = []
    if len(filtered_polygons) > 0 and len(labels) > 0:
        masks = create_mask(image, filtered_polygons, labels, fill_mask=True, use_color=False)
    else:
        import pdb; pdb.set_trace()
    return masks

def overlay_image(image, mask):
    lily_img = Image.open('./Lily.jpg').resize((image.width, image.height))
    plt.figure()
    plt.subplot(1, 2, 1)
    contour = torch.where((mask == 1), torch.zeros(1), torch.ones(1))
    tmp = torch.where((contour == 1.), image, transforms.ToTensor()(lily_img))
    plt.imshow(tmp.permute(1,2, 0))
    plt.title('car')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    tmp = mask.permute(1, 2, 0).cpu().detach().numpy() 
    plt.imshow(tmp)
    plt.title(f'mask{idx}')
    plt.axis('off')
    plt.savefig('testinf.png')

file_path = f'./data/nuscenes/train/samples/CAM_FRONT/n015-2018-10-08-15-44-23+0800__CAM_FRONT__1538985062362460.jpg' #n015-2018-11-21-19-58-31+0800__CAM_FRONT__1542801728362460.jpg'
image = Image.open(file_path)
orig_height = image.height
orig_width = image.width
height=9*24
width=16*24
image = image.resize((width, height))
masks = run_camou(image)

# idx = random.randint(0, mask.shape[0])
image = Image.open(file_path)
for idx in range(0, masks.shape[0]):
    curr_mask = masks[idx]

    plt.figure()
    plt.subplot(1, 2, 1)
    curr_mask = torch.nn.functional.interpolate(curr_mask.unsqueeze(dim=1), size=(orig_height, orig_width), mode='bicubic', align_corners=False)
    tmp = curr_mask.squeeze(dim=1).permute(1, 2, 0).cpu().detach().numpy() * transforms.ToTensor()(image).permute(1, 2, 0).numpy()
    tmp[tmp == 0] = 1
    plt.imshow(tmp)
    plt.title('car')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    tmp = curr_mask.squeeze(dim=1).permute(1, 2, 0).cpu().detach().numpy() 
    # zero_mask = (tmp == 0)
    # one_mask = (tmp == 1)
    # tmp[zero_mask] = 1
    # tmp[one_mask] = 0
    plt.imshow(tmp)
    plt.title('mask')
    plt.axis('off')
    plt.savefig(f'test{idx}.png')

overlay_image(image, curr_mask.squeeze(dim=1))