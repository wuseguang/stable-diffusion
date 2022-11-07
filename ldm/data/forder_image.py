import os
import numpy as np
from omegaconf import ListConfig
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import random
import albumentations
import cv2

class FolderData(Dataset):
    def __init__(self,
        root_dir,
        size,
        ext="png",
        split="train",
        random_crop=False,
        min_crop_f=0.5, 
        max_crop_f=1.
        ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.root_dir = root_dir
        self.split = split
        self.min_crop_f = min_crop_f
        self.max_crop_f = max_crop_f
        self.path_file = os.path.join(self.root_dir, split + ".list.txt")
        self.size = size
        self.center_crop = not random_crop
        self.image_rescaler = albumentations.SmallestMaxSize(max_size=size, interpolation=cv2.INTER_AREA)

        if not isinstance(ext, (tuple, list, ListConfig)):
            ext = [ext]

        if not os.path.exists(self.path_file):
            paths = []
            for e in ext:
                paths.extend(list(Path(self.root_dir).rglob(f"*.{e}")))
            paths = [str(x) for x in paths]
            total = len(paths)
            #train_count = int(total * 0.9)
            train_count = total - 100
            random.shuffle(paths)
            train_path_file = os.path.join(self.root_dir, "train.list.txt")
            val_path_file = os.path.join(self.root_dir, "val.list.txt")
            with open(train_path_file, 'w') as f:
                print("\n".join(paths[:train_count]), file=f)
            with open(val_path_file, 'w') as f:
                print("\n".join(paths[train_count:]), file=f)
        with open(self.path_file) as f:
            self.paths = [x.strip() for x in f.readlines()]

        if self.split == 'train':
            random.shuffle(self.paths)

    def __len__(self):
        return len(self.paths)
        #return min(len(self.paths), 1000)

    def __getitem__(self, index):
        data = {}
        filename = self.paths[index]

        image = Image.open(filename)

        if not image.mode == "RGB":
            image = image.convert("RGB")

        image = np.array(image).astype(np.uint8)

        min_side_len = min(image.shape[:2])
        if self.split == 'train':
            crop_side_len = min_side_len * np.random.uniform(self.min_crop_f, self.max_crop_f, size=None)
            crop_side_len = int(crop_side_len)
        else:
            crop_side_len = min_side_len

        if self.center_crop or self.split != 'train':
            self.cropper = albumentations.CenterCrop(height=crop_side_len, width=crop_side_len)

        else:
            self.cropper = albumentations.RandomCrop(height=crop_side_len, width=crop_side_len)

        image = self.cropper(image=image)["image"]
        image = self.image_rescaler(image=image)["image"]

        data["image"] = (image/127.5 - 1.0).astype(np.float32)

        return data