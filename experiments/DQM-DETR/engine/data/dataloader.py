"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 D-FINE authors. All Rights Reserved.
"""

import torch
import torch.utils.data as data
import torch.nn.functional as F
from torch.utils.data import default_collate

import torchvision
import torchvision.transforms.v2 as VT
from torchvision.transforms.v2 import functional as VF, InterpolationMode

import random
from functools import partial

from ..core import register
torchvision.disable_beta_transforms_warning()
from copy import deepcopy
from PIL import Image, ImageDraw
import os


__all__ = [
    'DataLoader',
    'BaseCollateFunction',
    'BatchImageCollateFunction',
    'PairedDegradationCollateFunction',
    'batch_image_collate_fn'
]


def _set_worker_threads(threads, worker_id):
    torch.set_num_threads(threads)


@register()
class DataLoader(data.DataLoader):
    __inject__ = ['dataset', 'collate_fn']

    def __init__(self, *args, worker_cpu_threads=1, **kwargs):
        num_workers = kwargs.get('num_workers', 0)
        if num_workers > 0 and kwargs.get('worker_init_fn') is None:
            kwargs['worker_init_fn'] = partial(_set_worker_threads, worker_cpu_threads)
        super().__init__(*args, **kwargs)

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for n in ['dataset', 'batch_size', 'num_workers', 'drop_last', 'collate_fn']:
            format_string += "\n"
            format_string += "    {0}: {1}".format(n, getattr(self, n))
        format_string += "\n)"
        return format_string

    def set_epoch(self, epoch):
        self._epoch = epoch
        self.dataset.set_epoch(epoch)
        self.collate_fn.set_epoch(epoch)

    @property
    def epoch(self):
        return self._epoch if hasattr(self, '_epoch') else -1

    @property
    def shuffle(self):
        return self._shuffle

    @shuffle.setter
    def shuffle(self, shuffle):
        assert isinstance(shuffle, bool), 'shuffle must be a boolean'
        self._shuffle = shuffle


@register()
def batch_image_collate_fn(items):
    """only batch image
    """
    return torch.cat([x[0][None] for x in items], dim=0), [x[1] for x in items]


class BaseCollateFunction(object):
    def set_epoch(self, epoch):
        self._epoch = epoch

    @property
    def epoch(self):
        return self._epoch if hasattr(self, '_epoch') else -1

    def __call__(self, items):
        raise NotImplementedError('')


def generate_scales(base_size, base_size_repeat, low_ratio=0.75, high_ratio=1.25):
    """Side lengths in pixels, multiples of 32; from low_ratio*base to base, then repeat base, then down to high_ratio*base."""
    low = int(base_size * low_ratio / 32) * 32
    high_end = int(base_size * high_ratio / 32) * 32
    low = min(low, base_size)
    high_end = max(high_end, base_size)
    scale_repeat = (base_size - low) // 32
    scales = [low + i * 32 for i in range(scale_repeat)]
    scales += [base_size] * base_size_repeat
    scale_repeat_high = (high_end - base_size) // 32
    scales += [high_end - i * 32 for i in range(scale_repeat_high)]
    return scales


@register() 
class BatchImageCollateFunction(BaseCollateFunction):
    def __init__(
        self, 
        stop_epoch=None, 
        ema_restart_decay=0.9999,
        base_size=640,
        base_size_repeat=None,
        mixup_prob=0.0,
        mixup_epochs=[0, 0],
        data_vis=False,
        vis_save='./vis_dataset/',
        scale_low_ratio=None,
        scale_high_ratio=None,
        gpu_augment=False,
    ) -> None:
        super().__init__()
        self.base_size = base_size
        if base_size_repeat is not None:
            lr = 0.75 if scale_low_ratio is None else float(scale_low_ratio)
            hr = 1.25 if scale_high_ratio is None else float(scale_high_ratio)
            self.scales = generate_scales(base_size, base_size_repeat, low_ratio=lr, high_ratio=hr)
        else:
            self.scales = None
        self.stop_epoch = stop_epoch if stop_epoch is not None else 100000000
        self.ema_restart_decay = ema_restart_decay
        # FIXME Mixup
        self.mixup_prob, self.mixup_epochs = mixup_prob, mixup_epochs
        self.gpu_augment = gpu_augment
        if self.mixup_prob > 0:
            self.data_vis, self.vis_save = data_vis, vis_save
            os.makedirs(self.vis_save, exist_ok=True) if self.data_vis else None
            print("     ### Using MixUp with Prob@{} in {} epochs ### ".format(self.mixup_prob, self.mixup_epochs))
        if stop_epoch is not None:
            print("     ### Multi-scale Training until {} epochs ### ".format(self.stop_epoch))
            print("     ### Multi-scales@ {} ###        ".format(self.scales))
        self.print_info_flag = True
        # self.interpolation = interpolation

    def apply_mixup(self, images, targets):
        """
        Applies Mixup augmentation to the batch if conditions are met.

        Args:
            images (torch.Tensor): Batch of images.
            targets (list[dict]): List of target dictionaries corresponding to images.

        Returns:
            tuple: Updated images and targets
        """
        # Log when Mixup is permanently disabled
        if self.epoch == self.mixup_epochs[-1] and self.print_info_flag:
            print(f"     ### Attention --- Mixup is closed after epoch@ {self.epoch} ###")
            self.print_info_flag = False

        # Apply Mixup if within specified epoch range and probability threshold
        if random.random() < self.mixup_prob and self.mixup_epochs[0] <= self.epoch < self.mixup_epochs[-1]:
            # Generate mixup ratio
            beta = round(random.uniform(0.45, 0.55), 6)

            # Mix images
            images = images.roll(shifts=1, dims=0).mul_(1.0 - beta).add_(images.mul(beta))

            # Prepare targets for Mixup
            shifted_targets = targets[-1:] + targets[:-1]
            updated_targets = deepcopy(targets)

            for i in range(len(targets)):
                # Combine boxes, labels, and areas from original and shifted targets
                updated_targets[i]['boxes'] = torch.cat([targets[i]['boxes'], shifted_targets[i]['boxes']], dim=0)
                updated_targets[i]['labels'] = torch.cat([targets[i]['labels'], shifted_targets[i]['labels']], dim=0)
                updated_targets[i]['area'] = torch.cat([targets[i]['area'], shifted_targets[i]['area']], dim=0)

                # Add mixup ratio to targets
                updated_targets[i]['mixup'] = torch.tensor(
                    [beta] * len(targets[i]['labels']) + [1.0 - beta] * len(shifted_targets[i]['labels']), 
                    dtype=torch.float32,
                    device=images.device
                    )
            targets = updated_targets

            if self.data_vis:
                for i in range(len(updated_targets)):
                    image_tensor = images[i]
                    image_tensor_uint8 = (image_tensor * 255).type(torch.uint8)
                    image_numpy = image_tensor_uint8.numpy().transpose((1, 2, 0))
                    pilImage = Image.fromarray(image_numpy)
                    draw = ImageDraw.Draw(pilImage)
                    print('mix_vis:', i, 'boxes.len=', len(updated_targets[i]['boxes']))
                    for box in updated_targets[i]['boxes']:
                        draw.rectangle([int(box[0]*640 - (box[2]*640)/2), int(box[1]*640 - (box[3]*640)/2), 
                                        int(box[0]*640 + (box[2]*640)/2), int(box[1]*640 + (box[3]*640)/2)], outline=(255,255,0))
                    pilImage.save(self.vis_save + str(i) + "_"+ str(len(updated_targets[i]['boxes'])) +'_out.jpg')

        return images, targets

    def apply_batch_augment(self, images, targets):
        images, targets = self.apply_mixup(images, targets)

        if self.scales is not None and self.epoch < self.stop_epoch:
            sz = random.choice(self.scales)
            images = F.interpolate(images, size=sz)
            if 'masks' in targets[0]:
                for tg in targets:
                    tg['masks'] = F.interpolate(tg['masks'], size=sz, mode='nearest')
                raise NotImplementedError('')

        return images, targets

    def __call__(self, items):
        images = torch.cat([x[0][None] for x in items], dim=0)
        targets = [x[1] for x in items]

        if not self.gpu_augment:
            images, targets = self.apply_batch_augment(images, targets)

        return images, targets


@register()
class PairedDegradationCollateFunction(BatchImageCollateFunction):
    def __init__(
        self,
        degradation_prob=1.0,
        min_ops=1,
        max_ops=2,
        low_light_range=[0.45, 0.85],
        contrast_range=[0.55, 0.9],
        noise_std_range=[0.01, 0.04],
        blur_kernel=5,
        blur_sigma_range=[0.3, 1.2],
        fog_alpha_range=[0.08, 0.22],
        object_aware=False,
        foreground_strength=0.75,
        background_strength=1.25,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.degradation_prob = degradation_prob
        self.min_ops = min_ops
        self.max_ops = max_ops
        self.low_light_range = low_light_range
        self.contrast_range = contrast_range
        self.noise_std_range = noise_std_range
        self.blur_kernel = blur_kernel
        self.blur_sigma_range = blur_sigma_range
        self.fog_alpha_range = fog_alpha_range
        self.object_aware = object_aware
        self.foreground_strength = foreground_strength
        self.background_strength = background_strength

    def _uniform(self, value_range):
        return random.uniform(float(value_range[0]), float(value_range[1]))

    def _apply_low_light(self, image):
        return image * self._uniform(self.low_light_range)

    def _apply_contrast(self, image):
        mean = image.mean(dim=(-2, -1), keepdim=True)
        return (image - mean) * self._uniform(self.contrast_range) + mean

    def _apply_noise(self, image):
        return image + torch.randn_like(image) * self._uniform(self.noise_std_range)

    def _apply_blur(self, image):
        kernel = int(self.blur_kernel)
        kernel = kernel if kernel % 2 == 1 else kernel + 1
        sigma = self._uniform(self.blur_sigma_range)
        return VF.gaussian_blur(image, kernel_size=[kernel, kernel], sigma=[sigma, sigma])

    def _apply_fog(self, image):
        alpha = self._uniform(self.fog_alpha_range)
        veil = torch.full_like(image, 0.75)
        return image * (1.0 - alpha) + veil * alpha

    def _foreground_mask(self, target, height, width, device, dtype):
        mask = torch.zeros(1, height, width, device=device, dtype=dtype)
        boxes = target.get('boxes') if target is not None else None
        if boxes is None or boxes.numel() == 0:
            return mask

        boxes = boxes.to(device=device, dtype=dtype)
        x1 = (boxes[:, 0] - boxes[:, 2] * 0.5) * width
        y1 = (boxes[:, 1] - boxes[:, 3] * 0.5) * height
        x2 = (boxes[:, 0] + boxes[:, 2] * 0.5) * width
        y2 = (boxes[:, 1] + boxes[:, 3] * 0.5) * height

        for left, top, right, bottom in zip(x1, y1, x2, y2):
            left = int(torch.floor(left).clamp(0, width).item())
            top = int(torch.floor(top).clamp(0, height).item())
            right = int(torch.ceil(right).clamp(0, width).item())
            bottom = int(torch.ceil(bottom).clamp(0, height).item())
            if right > left and bottom > top:
                mask[:, top:bottom, left:right] = 1
        return mask

    def _apply_object_aware_strength(self, clean, degraded, target):
        _, height, width = clean.shape
        mask = self._foreground_mask(target, height, width, clean.device, clean.dtype)
        strength = torch.where(
            mask > 0,
            torch.as_tensor(self.foreground_strength, device=clean.device, dtype=clean.dtype),
            torch.as_tensor(self.background_strength, device=clean.device, dtype=clean.dtype),
        )
        return clean + (degraded - clean) * strength

    def _degrade_one(self, image, target=None):
        if random.random() >= self.degradation_prob:
            return image

        clean = image
        ops = [
            self._apply_low_light,
            self._apply_contrast,
            self._apply_noise,
            self._apply_blur,
            self._apply_fog,
        ]
        num_ops = random.randint(int(self.min_ops), int(self.max_ops))
        for op in random.sample(ops, k=min(num_ops, len(ops))):
            image = op(image)
        if self.object_aware:
            image = self._apply_object_aware_strength(clean, image, target)
        return image.clamp(0.0, 1.0)

    def __call__(self, items):
        images, targets = super().__call__(items)
        degraded = torch.stack([self._degrade_one(image.clone(), target) for image, target in zip(images, targets)], dim=0)
        return {'clean': images, 'degraded': degraded, 'targets': targets}
