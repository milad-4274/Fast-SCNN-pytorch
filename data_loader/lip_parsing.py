"""Look into Person Dataset"""
import os
import torch
import random
import numpy as np

from PIL import Image, ImageOps, ImageFilter
# from core.data.dataloader.segbase import SegmentationDataset
import torch.utils.data as data


class LIPSegmentation(data.Dataset):
    """Look into person parsing dataset """

    BASE_DIR = 'LIP'
    NUM_CLASS = 20


    def __init__(self, root=os.path.abspath('datasets/LIP'), split='train', mode=None, transform=None,
    base_size=520, crop_size=480, **kwargs):
        print("initialize LIPSegmantation dataset and path is",root)
        super(LIPSegmentation, self).__init__()
        self.mode = mode if mode is not None else split
        self.crop_size = crop_size
        self.base_size = base_size
        self.transform = transform
        _trainval_image_dir = os.path.join(root, 'TrainVal_images')#T
        _testing_image_dir = os.path.join(root, 'Testing_images')
        _trainval_mask_dir = os.path.join(root, 'TrainVal_parsing_annotations')
        if split == 'train':
            _image_dir = os.path.join(_trainval_image_dir, 'train_images')
            _mask_dir = os.path.join(_trainval_mask_dir, 'train_segmentations')
            _split_f = os.path.join(_trainval_image_dir, 'train_id.txt')
        elif split == 'val':
            _image_dir = os.path.join(_trainval_image_dir, 'val_images')
            _mask_dir = os.path.join(_trainval_mask_dir, 'val_segmentations')
            _split_f = os.path.join(_trainval_image_dir, 'val_id.txt')
        elif split == 'test':
            _image_dir = os.path.join(_testing_image_dir, 'testing_images')
            _split_f = os.path.join(_testing_image_dir, 'test_id.txt')
        else:
            raise RuntimeError('Unknown dataset split.')

        self.images = []
        self.masks = []
        with open(os.path.join(_split_f), 'r') as lines:
            for line in lines:
                _image = os.path.join(_image_dir, line.rstrip('\n') + '.jpg')
                assert os.path.isfile(_image)
                self.images.append(_image)
                if split != 'test':
                    _mask = os.path.join(_mask_dir, line.rstrip('\n') + '.png')
                    print("mask",_mask)
                    assert os.path.isfile(_mask)
                    self.masks.append(_mask)

        if split != 'test':
            assert (len(self.images) == len(self.masks))
        print('Found {} {} images in the folder {}'.format(len(self.images), split, root))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            img = self._img_transform(img)
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        mask = Image.open(self.masks[index])
        # synchronized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            img, mask = self._img_transform(img), self._mask_transform(mask)
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)

        return img, mask, os.path.basename(self.images[index])

    def __len__(self):
        return len(self.images)

    def _mask_transform(self, mask):
        target = np.array(mask).astype('int32')
        return torch.from_numpy(target).long()

    @property
    def classes(self):
        """Category name."""
        return ('background', 'hat', 'hair', 'glove', 'sunglasses', 'upperclothes',
                'dress', 'coat', 'socks', 'pants', 'jumpsuits', 'scarf', 'skirt',
                'face', 'leftArm', 'rightArm', 'leftLeg', 'rightLeg', 'leftShoe',
                'rightShoe')

    @property
    def num_class(self):
        """Number of categories."""
        return self.NUM_CLASS

    def _sync_transform(self, img, mask):
            # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        # gaussian blur as in PSP
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _img_transform(self, img):
        return np.array(img)
if __name__ == '__main__':
    dataset = LIPSegmentation(base_size=280, crop_size=256)
