import torchvision.transforms as transforms
import numpy as np
import math

class RandomRotate(object):
    def __init__(self, p, angle):
        self.angle = angle
        self.p = p

    def __call__(self, camou2):
        if np.random.rand(1)>self.p:
            camou3 = transforms.functional.rotate(camou2, self.angle)
        else:
            camou3 = camou2
        return camou3

class RandomHorizontalFlip(object):
    def __init__(self, p):
        self.flip = transforms.RandomHorizontalFlip(p=p)

    def __call__(self, data):
        return self.flip(data)

class RandomVerticalFlip(object):
    def __init__(self, p):
        self.flip = transforms.RandomVerticalFlip(p=p)

    def __call__(self, data):
        return self.flip(data)

class RandomCrop(object):
    def __init__(self, size):
        self.flip = transforms.RandomCrop(size)

    def __call__(self, data):
        return self.flip(data).permute(0, 2, 3, 1)


class Stack(object):

    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        if img_group[0].mode == 'L':
            return np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif img_group[0].mode == 'RGB':
            if self.roll:
                return np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
            else:
                rst = np.concatenate(img_group, axis=2)
                # plt.imshow(rst[:,:,3:6])
                # plt.show()
                return rst

class Repeat(object):
    def __init__(self, t=6):
        self.t = 6

    def __call__(self, data):
        return data.repeat(self.t*self.t, 1, 1, 1)

class Tile(object):
    def __init__(self, t=6):
        self.t = 6

    def __call__(self, data):
        B, C, H, W = data.shape
        assert math.sqrt(B) == self.t 
        x1 = data.view(self.t, self.t, C, H, W)
        x1 = x1.permute(0, 2, 1, 3, 4)
        x1 = x1.reshape(self.t, C, 6*H, W)
        x2 = x1.permute(1, 2, 0, 3)
        x2 = x2.reshape(C, H*self.t, W*self.t)
        return x2.unsqueeze(0)

def get_augmentation(size):
    unique = transforms.Compose(
        [
            Repeat(t=6),
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            RandomRotate(p=0.5, angle=90),
            Tile(t=6),
            RandomCrop(size)
        ]
    )

    return transforms.Compose([unique])

def tex_trans(camou, num_rows=6, num_cols=6, size=4096):
    """
    Flip, rotate, and crop the camouflage texture
    """
    horizontal_flip = transforms.RandomHorizontalFlip(p=0.5)
    vertical_flip = transforms.RandomVerticalFlip(p=0.5)
    random_rotate = RandomRotate(p=0.5, angle=90)
    random_crop = transforms.RandomCrop(size)

    camou_column = []
    for i in range(num_cols):
        camou_row_list = []
        for j in range(num_rows):
            camou1 = horizontal_flip(camou.permute(0, 3, 1, 2)[0])
            camou2 = vertical_flip(camou1)
            camou3 = random_rotate(camou2)
            camou_row_list.append(camou3)
        camou_row = torch.cat(tuple(camou_row_list), 1)
        camou_column.append(camou_row)
    camou_full = torch.cat(tuple(camou_column), 2).unsqueeze(0)
    camou_crop = random_crop(camou_full).permute(0, 2, 3, 1)
    return camou_crop

if __name__ == '__main__':
    import torch
    from copy import deepcopy
    size = (368, 1056)
    aug = get_augmentation(size)

    device = 'cpu'

    resolution = 8
    expand_kernel = torch.nn.ConvTranspose2d(3, 3, resolution, stride=resolution, padding=0).to(device)
    expand_kernel.weight.data.fill_(0)
    expand_kernel.bias.data.fill_(0)
    for i in range(3):
        expand_kernel.weight[i, i, :, :].data.fill_(1)

    camou_para = torch.rand([1, 1056//resolution, 1056//resolution, 3]).float().to(device)
    camou_para.requires_grad_(True)
    begin_para = deepcopy(camou_para)
    camou_para1 = expand_kernel(camou_para.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

    import matplotlib.pyplot as plt

    t0 = aug(camou_para1.permute(0, 3, 1, 2)[0])
    t1 = tex_trans(camou_para1, size=size)

    plt.imshow(t0[0].detach().numpy())
    plt.axis('off')
    plt.savefig('test0.png')
    
    plt.imshow(t1[0].detach().numpy())
    plt.axis('off')
    plt.savefig('test1.png')