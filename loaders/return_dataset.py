import os
import torch
import cv2
from albumentations.pytorch import ToTensorV2
import albumentations as A

from loaders.dataset import Imagelists_VISDA
from loaders.dataset import PairedAugDataset

def get_data_transforms(args, transform='train'):
    crop_size = 224
    load_size = int(crop_size * 1.15)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    data_transforms = {
            'train': A.Compose([
                A.augmentations.geometric.resize.Resize(width=load_size, height=load_size),
                A.HorizontalFlip(),
                A.RandomCrop(width=crop_size, height=crop_size),
                A.Normalize(mean, std),
                ToTensorV2(),
            ]),
            'test': A.Compose([
                A.augmentations.geometric.resize.Resize(width=load_size, height=load_size),
                A.CenterCrop(width=crop_size, height=crop_size),
                A.Normalize(mean, std),
                ToTensorV2(),
            ]),
            'randaug': A.Compose([
                A.augmentations.geometric.resize.Resize(width=load_size, height=load_size),
                A.SomeOf(transforms=[
                    A.ColorJitter(brightness=0.75, contrast=0, saturation=0.95, hue=0, always_apply=True),
                    A.ColorJitter(brightness=0, contrast=0, saturation=0.99, hue=0, always_apply=True),
                    A.ColorJitter(brightness=0, contrast=0.95, saturation=0, hue=0, always_apply=True),
                    A.Equalize(mode='pil', by_channels=True, always_apply=True),
                    A.Posterize(num_bits=[4, 8], always_apply=True),
                    A.Sharpen(alpha=[0, 0.95], lightness=[0.5, 1.0], always_apply=True),
                    A.Solarize(threshold=[0, 256], always_apply=True),
                    A.Rotate(limit=[-30,30], always_apply=True),
                    A.Affine(shear={'x':[-20, 20]}, always_apply=True),
                    A.Affine(shear={'y':[-20, 20]}, always_apply=True),
                    A.ShiftScaleRotate(shift_limit_x=0.3, shift_limit=0, scale_limit=0, rotate_limit=0, border_mode=cv2.BORDER_CONSTANT, always_apply=True),
                    A.ShiftScaleRotate(shift_limit_y=0.3, shift_limit=0, scale_limit=0, rotate_limit=0, border_mode=cv2.BORDER_CONSTANT, always_apply=True),
                    A.HorizontalFlip(p=0.01) # instead of identity
                ], n = 3, replace=False),
                A.CoarseDropout(min_holes=1, max_holes=1,
                                min_height=1, max_height=int(load_size * 0.5),
                                min_width=1, max_width=int(load_size * 0.5),
                                fill_value=[125, 123, 114],
                                always_apply=True),
                A.HorizontalFlip(p=0.5),
                A.RandomCrop(width=crop_size, height=crop_size),
                A.Normalize(mean, std),
                ToTensorV2(),
            ])
        }

    if isinstance(transform, str):
        return data_transforms[transform]
    else:
        return transform

def get_loader(args, dataset, shuffle=True, bs=None, nw=None, dl=True, sampler=None):
    if(bs is None):
        bs = args.bs

    if(nw is None):
        nw = args.num_workers

    loader = torch.utils.data.DataLoader(dataset, batch_size=min(bs, len(dataset)),
                                         sampler=sampler,
                                         num_workers=nw, shuffle=shuffle, drop_last=dl, pin_memory=True)
    return loader

def get_dataset(args, split, transform='train', paired=False, transform2=None, isalbu=True):
    base_path = './data/txt/{}'.format(args.dataset)
    root = './data/{}/'.format(args.dataset)
    domain = args.source if 'source' in split else args.target
    appendix = '{}.txt'.format(split.split('_')[-1])
    image_set_file = "{}/{}_{}".format(base_path, domain, appendix)
    assert(os.path.exists(image_set_file)), "image set file \'{}\' not exist".format(image_set_file)

    if paired:
        transform1 = transform
        dataset = PairedAugDataset(image_set_file,
                                    root = root,
                                    transform = get_data_transforms(args, transform1),
                                    transform2 = get_data_transforms(args, transform2))
    else:
        dataset = Imagelists_VISDA(image_set_file, root=root, transform=get_data_transforms(args, transform), isalbu=isalbu)

    return dataset