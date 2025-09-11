import albumentations as A
from albumentations.pytorch import ToTensorV2
def get_train_transforms(img_size=256, task="seg"):
    image_mask_aug = [
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
    ]
    image_only_aug = [
        A.RandomBrightnessContrast(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
    if task in ["seg", "joint"]:
        return A.Compose(image_mask_aug + [
            A.Normalize(mean=(0.5, 0.5, 0.5),
                        std=(0.5, 0.5, 0.5)),
            ToTensorV2()],
            additional_targets={"mask": "mask"})
    elif task == "cls":
        return A.Compose(image_mask_aug + image_only_aug)
    else:
        raise ValueError(f"Unknown task: {task}")
def get_val_transforms(img_size=256, task="seg"):
    if task in ["seg", "joint"]:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.5, 0.5, 0.5),
                        std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ])
    elif task == "cls":
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        raise ValueError(f"Unknown task: {task}")