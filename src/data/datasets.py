import os
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
CLASS_MAP = {
    "glioma": 0,
    "meningioma": 1,
    "no_tumor": 2,
    "pituitary": 3,
}
class BriscDataset(Dataset):
    def __init__(self, data_root, split="train", task="seg", img_size=256):
        self.task = task
        self.img_size = img_size
        if task in ["seg", "joint"]:
            self.img_dir = os.path.join(data_root, "segmentation_task", split, "images")
            self.mask_dir = os.path.join(data_root, "segmentation_task", split, "masks")
            self.img_files = sorted(os.listdir(self.img_dir))
        elif task == "cls":
            self.img_dir = os.path.join(data_root, "classification_task", split)
            self.img_files = []
            for cls_name, cls_idx in CLASS_MAP.items():
                cls_dir = os.path.join(self.img_dir, cls_name)
                for f in os.listdir(cls_dir):
                    self.img_files.append((os.path.join(cls_dir, f), cls_idx))
        if split == "train":
            self.img_transform = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ColorJitter(p=0.3),
                A.Normalize(mean=(0.5, 0.5, 0.5),
                            std=(0.5, 0.5, 0.5)),
                ToTensorV2(),
            ])
            self.img_mask_transform = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                ToTensorV2(),
            ])
            self.image_mask_transform = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
            ], additional_targets={"mask": "mask"})
            self.normalize_transform = A.Compose([
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ToTensorV2(),
            ])
        else:
            self.img_transform = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=(0.5, 0.5, 0.5),
                            std=(0.5, 0.5, 0.5)),
                ToTensorV2(),
            ])
            self.img_mask_transform = A.Compose([
                A.Resize(img_size, img_size),
                ToTensorV2(),
            ])
            self.image_mask_transform = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
            ], additional_targets={"mask": "mask"})
            self.normalize_transform = A.Compose([
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ToTensorV2(),
            ])
    def __len__(self):
        return len(self.img_files)
    def __getitem__(self, idx):
        if self.task in ["seg", "joint"]:
            img_name = self.img_files[idx]
            img_path = os.path.join(self.img_dir, img_name)
            mask_path = os.path.join(self.mask_dir, img_name.replace(".jpg", ".png"))
            image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype("float32")
            if self.task == "seg":
                augmented = self.image_mask_transform(image=image, mask=mask)
                image_np = augmented["image"]
                mask_final = augmented["mask"]
                image_final = self.normalize_transform(image=image_np)["image"]
                mask_final = torch.from_numpy(mask_final).unsqueeze(0).float()
                return image_final, mask_final
            elif self.task == "joint":
                augmented = self.image_mask_transform(image=image, mask=mask)
                image_np = augmented["image"]
                mask_np = augmented["mask"]
                img_norm = self.normalize_transform(image=image_np)["image"]
                mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).float()
                label_map = {
                    "gl": 0,
                    "me": 1,
                    "nt": 2,
                    "pi": 3,
                }
                label = None
                for key, cls_idx in label_map.items():
                    if f"_{key}_" in img_name.lower():
                        label = cls_idx
                        break
                if label is None:
                    raise ValueError(f"Could not determine class label from filename: {img_name}")
                return img_norm, mask_tensor, torch.tensor(label, dtype=torch.long)
        elif self.task == "cls":
            img_path, label = self.img_files[idx]
            image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            aug = self.img_transform(image=image)
            return aug["image"], torch.tensor(label, dtype=torch.long)