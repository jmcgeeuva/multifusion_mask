import numpy as np
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from PIL import Image
import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

def ade_palette():
    """ADE20K palette that maps each class to RGB values."""
    return [
        [120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50], [4, 200, 3],
        [120, 120, 80], [140, 140, 140], [204, 5, 255], [230, 230, 230], [4, 250, 7],
        [224, 5, 255], [235, 255, 7], [150, 5, 61], [120, 120, 70], [8, 255, 51],
        [255, 6, 82], [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
        [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255], [255, 7, 71],
        [255, 9, 224], [9, 7, 230], [220, 220, 220], [255, 9, 92], [112, 9, 255],
        [8, 255, 214], [7, 255, 224], [255, 6, 245], [255, 0, 245], [255, 9, 128],
        [128, 128, 128], [102, 0, 102], [200, 102, 0], [51, 51, 153], [255, 102, 255],
        [255, 102, 102], [153, 255, 153], [255, 153, 153], [102, 255, 102], [153, 102, 255],
        [153, 153, 255], [255, 153, 255], [255, 255, 102], [153, 255, 255], [102, 153, 255],
        [255, 102, 153], [153, 255, 102], [102, 153, 153], [153, 102, 102], [255, 153, 102],
        [102, 255, 153], [153, 102, 153], [102, 153, 102], [255, 102, 102], [102, 153, 102],
        [0, 128, 0], [128, 0, 0], [0, 0, 128], [128, 0, 128], [0, 128, 128],
        [128, 128, 128], [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
        [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0],
        [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128], [128, 64, 128],
        [0, 192, 128], [128, 192, 128], [64, 64, 0], [192, 64, 0], [64, 192, 0],
        [192, 192, 0], [64, 64, 128], [192, 64, 128], [64, 192, 128], [192, 192, 128],
        [0, 0, 64], [128, 0, 64], [0, 128, 64], [128, 128, 64], [0, 0, 192],
        [128, 0, 192], [0, 128, 192], [128, 128, 192], [64, 0, 64], [192, 0, 64],
        [64, 128, 64], [192, 128, 64], [64, 0, 192], [192, 0, 192], [64, 128, 192],
        [192, 128, 192], [0, 64, 64], [128, 64, 64], [0, 192, 64], [128, 192, 64],
        [0, 64, 192], [128, 64, 192], [0, 192, 192], [128, 192, 192], [64, 64, 64],
        [192, 64, 64], [64, 192, 64], [192, 192, 64], [64, 64, 192], [192, 64, 192],
        [64, 192, 192], [192, 192, 192], [32, 0, 0], [160, 0, 0], [32, 128, 0],
        [160, 128, 0], [32, 0, 128], [160, 0, 128], [32, 128, 128], [160, 128, 128],
        [96, 0, 0], [224, 0, 0], [96, 128, 0], [224, 128, 0], [96, 0, 128],
        [224, 0, 128], [96, 128, 128], [224, 128, 128], [32, 64, 0], [160, 64, 0],
        [32, 192, 0], [160, 192, 0], [32, 64, 128], [160, 64, 128], [32, 192, 128],
        [160, 192, 128], [96, 64, 0], [224, 64, 0], [96, 192, 0], [224, 192, 0],
        [96, 64, 128], [224, 64, 128], [96, 192, 128], [224, 192, 128], [32, 0, 64],
        [160, 0, 64], [32, 128, 64], [160, 128, 64], [32, 0, 192], [160, 0, 192],
        [32, 128, 192], [160, 128, 192], [96, 0, 64], [224, 0, 64], [96, 128, 64],
        [224, 128, 64], [96, 0, 192], [224, 0, 192], [96, 128, 192], [224, 128, 192],
        [32, 64, 64], [160, 64, 64], [32, 192, 64], [160, 192, 64], [32, 64, 192],
        [160, 64, 192], [32, 192, 192], [160, 192, 192], [96, 64, 64], [224, 64, 64],
        [96, 192, 64], [224, 192, 64], [96, 64, 192], [224, 64, 192], [96, 192, 192],
        [224, 192, 192]
    ]


def segment_image(input_image, alpha: float = 0.5) -> Image.Image:
    """
    Performs semantic segmentation on an image and returns an overlay mask.

    Args:
        input_image (str or PIL.Image): Path to an image file, URL, or a PIL Image instance.
        alpha (float): Blending factor for overlay. 0 = only original, 1 = only mask.

    Returns:
        PIL.Image: The original image blended with the segmentation mask.
    """
    # Load image
    if isinstance(input_image, str):
        if input_image.startswith(('http://', 'https://')):
            resp = requests.get(input_image)
            img = Image.open(BytesIO(resp.content)).convert("RGB")
        else:
            img = Image.open(input_image).convert("RGB")
    elif isinstance(input_image, Image.Image):
        img = input_image.convert("RGB")
    else:
        raise ValueError("Unsupported input type. Provide a file path, URL, or PIL.Image.")

    # Preprocess and forward pass
    pixel_values = processor(img, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        outputs = model(pixel_values)

    # Post-process to get mask
    seg_map = processor.post_process_semantic_segmentation(
        outputs, target_sizes=[img.size[::-1]]
    )[0].cpu().numpy()

    id2label = model.config.id2label

    # Create color mask
    palette = np.array(ade_palette(), dtype=np.uint8)
    color_mask = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        if label >= len(id2label.keys()):
            break
        try:
            if id2label[label] in ['car', 'bus', 'person']:
                color_mask[seg_map == label] = color
        except:
            import pdb; pdb.set_trace()

    # Blend original and mask
    orig_arr = np.array(img)
    overlay = (orig_arr * (1 - alpha) + color_mask * alpha).astype(np.uint8)
    return Image.fromarray(overlay)

# Initialize model and processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "nvidia/segformer-b5-finetuned-ade-640-640"
processor = SegformerImageProcessor(do_resize=False)
model = SegformerForSemanticSegmentation.from_pretrained(model_name)
model.to(device)

# Example usage:
# url = "https://i.pinimg.com/736x/f7/5a/f2/f75af26820b50c24600f50f3998eb02f.jpg"
# file_path = f'./data/nuscenes/train/samples/CAM_FRONT/n015-2018-10-08-15-44-23+0800__CAM_FRONT__1538985062362460.jpg'
file_path = f'./data/nuscenes/train/samples/CAM_FRONT/n015-2018-11-21-19-58-31+0800__CAM_FRONT__1542801728362460.jpg'
image = Image.open(file_path)
result = segment_image(image, alpha=1)

import matplotlib.pyplot as plt
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(result)
plt.subplot(1, 2, 2)
plt.imshow(image)
plt.axis('off')
plt.savefig('test.png')