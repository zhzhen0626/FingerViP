import numbers
import math
import warnings
from typing import Tuple, List, Optional
import torch
from torch import Tensor
import torchvision.transforms.functional as F

# from https://github.com/pytorch/vision/blob/4521f6d152875974e317fa247a633e9ad1ea05c8/torchvision/transforms/transforms.py#L35https://github.com/pytorch/vision/blob/4521f6d152875974e317fa247a633e9ad1ea05c8/torchvision/transforms/transforms.py#L35

class RandomErasing(torch.nn.Module):
    """ Randomly selects a rectangle region in an image and erases its pixels.
    'Random Erasing Data Augmentation' by Zhong et al. See https://arxiv.org/abs/1708.04896

    Args:
         p: probability that the random erasing operation will be performed.
         scale: range of proportion of erased area against input image.
         ratio: range of aspect ratio of erased area.
         value: erasing value. Default is 0. If a single int, it is used to
            erase all pixels. If a tuple of length 3, it is used to erase
            R, G, B channels respectively.
            If a str of 'random', erasing each pixel with random values.
         inplace: boolean to make this transform inplace. Default set to False.

    Returns:
        Erased Image.

    Example:
        >>> transform = transforms.Compose([
        >>>   transforms.RandomHorizontalFlip(),
        >>>   transforms.ToTensor(),
        >>>   transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        >>>   transforms.RandomErasing(),
        >>> ])
    """

    def __init__(self, p=0.5, scale=(0., 0.33), ratio=(0.3, 3.3), value=0, inplace=False):
        super().__init__()
        if not isinstance(value, (numbers.Number, str, tuple, list)):
            raise TypeError("Argument value should be either a number or str or a sequence")
        if isinstance(value, str) and value != "random":
            raise ValueError("If value is str, it should be 'random'")
        if not isinstance(scale, (tuple, list)):
            raise TypeError("Scale should be a sequence")
        if not isinstance(ratio, (tuple, list)):
            raise TypeError("Ratio should be a sequence")
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")
        if scale[0] < 0 or scale[1] > 1:
            raise ValueError("Scale should be between 0 and 1")
        if p < 0 or p > 1:
            raise ValueError("Random erasing probability should be between 0 and 1")

        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
        self.inplace = inplace

    @staticmethod
    def get_params(
            img: Tensor, scale: Tuple[float, float], ratio: Tuple[float, float], value: Optional[List[float]] = None
    ) -> Tuple[int, int, int, int, Tensor]:
        """Get parameters for ``erase`` for a random erasing.

        Args:
            img (Tensor): Tensor image to be erased. (N, C, H, W)
            scale (tuple or list): range of proportion of erased area against input image.
            ratio (tuple or list): range of aspect ratio of erased area.
            value (list, optional): erasing value. If None, it is interpreted as "random"
                (erasing each pixel with random values). If ``len(value)`` is 1, it is interpreted as a number,
                i.e. ``value[0]``.

        Returns:
            tuple: params (i, j, h, w, v) to be passed to ``erase`` for random erasing.
        """
        img_c, img_h, img_w = img.shape[-3], img.shape[-2], img.shape[-1]
        area = img_h * img_w

        for _ in range(10):
            erase_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.empty(1).uniform_(ratio[0], ratio[1]).item()

            h = int(round(math.sqrt(erase_area * aspect_ratio)))
            w = int(round(math.sqrt(erase_area / aspect_ratio)))
            if not (h < img_h and w < img_w):
                continue

            if value is None:
                v = torch.empty([img_c, h, w], dtype=torch.float32).normal_()
            else:
                v = torch.tensor(value)[:, None, None]

            i = torch.randint(0, img_h - h + 1, size=(1, )).item()
            j = torch.randint(0, img_w - w + 1, size=(1, )).item()
            return i, j, h, w, v

        # Return original image
        return 0, 0, img_h, img_w, img

    def forward(self, img):
        """
        Args:
            img (Tensor): Tensor image to be erased. (N, C, H, W)

        Returns:
            img (Tensor): Erased Tensor image.
        """
        if torch.rand(1) < self.p:

            # cast self.value to script acceptable type
            if isinstance(self.value, (int, float)):
                value = [self.value, ]
            elif isinstance(self.value, str):
                value = None
            elif isinstance(self.value, tuple):
                value = list(self.value)
            else:
                value = self.value

            if value is not None and not (len(value) in (1, img.shape[-3])):
                raise ValueError(
                    "If value is a sequence, it should have either a single value or "
                    "{} (number of input channels)".format(img.shape[-3])
                )
            erased_img = []
            for image in img:
                x, y, h, w, v = self.get_params(image, scale=self.scale, ratio=self.ratio, value=value)
                erased_img.append(F.erase(image, x, y, h, w, v, self.inplace))
            return torch.stack(erased_img)
            # x, y, h, w, v = self.get_params(img, scale=self.scale, ratio=self.ratio, value=value)
            # return F.erase(img, x, y, h, w, v, self.inplace)
        return img