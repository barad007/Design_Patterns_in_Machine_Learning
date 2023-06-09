import torch
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as T


class TrainTransforms:
    def __init__(
        self,
        *,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation=InterpolationMode.BILINEAR,
        hflip_prob=0.5,
        random_erase_prob=0.25,
        resize=256,
        crop_size=224,
    ):
        trans = [T.Resize(resize, interpolation=interpolation), T.CenterCrop(crop_size)]
        random_transforms = [
             T.AutoAugment(T.AutoAugmentPolicy.IMAGENET),
             T.ColorJitter(brightness=.5, hue=.3),
             T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
             T.RandomPosterize(bits=2),
             T.RandomSolarize(threshold=192.0),
             T.RandomAdjustSharpness(sharpness_factor=2),
             T.RandomAutocontrast(),
             T.RandomEqualize(),
             T.AugMix(),
             T.RandomInvert()
        ]
        trans.append(T.RandomChoice(transforms=random_transforms))
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))

        trans.extend(
            [
                T.PILToTensor(),
                T.ConvertImageDtype(torch.float),
                T.Normalize(mean=mean, std=std),
            ]
        )
        if random_erase_prob > 0:
            trans.append(T.RandomErasing(p=random_erase_prob))
        self.transforms = T.Compose(trans)

    def __call__(self, img):
        return self.transforms(img)


class ValidationTransforms:
    def __init__(
        self,
        *,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation=InterpolationMode.BILINEAR,
        resize=256,
        crop_size=224,
    ):
        trans = [
            T.Resize(resize, interpolation=interpolation),
            T.CenterCrop(crop_size),
            T.PILToTensor(),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean=mean, std=std),
        ]
        self.transforms = T.Compose(trans)

    def __call__(self, img):
        return self.transforms(img)

