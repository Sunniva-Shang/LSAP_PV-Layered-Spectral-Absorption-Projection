from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
import random
import os
import torch.nn as nn
import torch
from torchvision import transforms


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif", "bmp"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class VeinDataset(Dataset):
    def __init__(self, resolution, image_paths, cond_paths, classes=None, shard=0, num_shards=1):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_conds = cond_paths[shard:][::num_shards]
        self.local_classes = classes 
        resize_ratio = 0.1
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ColorJitter(brightness=0.01, contrast=0.01),
            transforms.Resize((int(resolution *(1 + resize_ratio)),int(resolution *(1 + resize_ratio)))),
        ])
        self.transform2 = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((int(resolution *(1 + resize_ratio)),int(resolution *(1 + resize_ratio)))),

        ]) 

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        imagepath = self.local_images[idx]
        condpath = self.local_conds[idx]
        if self.local_classes:
            sch = imagepath.split("/")[-2].split('_')[0]

        with bf.BlobFile(imagepath, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()

        with bf.BlobFile(condpath, "rb") as f:
            pil_cond = Image.open(f)
            pil_cond.load()

        pil_image = self.transform(pil_image)
        pil_cond = self.transform2(pil_cond)

        crop_y = random.randrange(pil_image.size[0] - self.resolution + 1)
        crop_x = random.randrange(pil_image.size[0] - self.resolution + 1)
        arr_image = random_crop_arr_with_point(pil_image, self.resolution, crop_x, crop_y)
        arr_cond = random_crop_arr_with_point(pil_cond, self.resolution, crop_x, crop_y)
        
        # norm
        arr_image = arr_image.astype(np.float32) / 127.5 - 1
        arr_cond = arr_cond.astype(np.float32) / 127.5 - 1
        out_dict = {}

        if self.local_classes:
            if sch =='casia':
                out_dict["y"] = np.array(0, dtype=np.int64)
            elif sch == 'HFUT':
                out_dict["y"] = np.array(1, dtype=np.int64)
            elif sch == 'polyu':
                out_dict["y"] = np.array(2, dtype=np.int64)
            elif sch == 'TongJi':
                out_dict["y"] = np.array(3, dtype=np.int64)
            else:
                print('The dataset category names are incorrect. Please modify them to CASIA, HFUT, PolyU, and Tongji.')

        out_dict["cond"] = np.transpose(arr_cond[:, :, None], [2, 0, 1])

        return np.transpose(arr_image[:, :, None], [2, 0, 1]), out_dict
        


class CondDataset(Dataset):
    def __init__(self, resolution, cond_paths, classes=None, shard=0, num_shards=1):
        super().__init__()
        self.resolution = resolution
        self.local_conds = cond_paths[shard:][::num_shards]
        self.local_classes = classes
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((int(resolution),int(resolution))),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5), 
        ]) 

    def __len__(self):
        return len(self.local_conds)

    def __getitem__(self, idx):
        condpath = self.local_conds[idx]
        if self.local_classes:   
            sch = condpath.split("/")[-2].split('_')[-1]

        with bf.BlobFile(condpath, "rb") as f:
            pil_cond = Image.open(f)
            pil_cond.load()
        arr_cond = self.transform(pil_cond).numpy()

        out_dict = {}
        if self.local_classes:
            out_dict["y"] = np.array(int(sch), dtype=np.int64)
       
        out_dict["cond"] = arr_cond 
        out_dict["path"] = condpath
        return out_dict 


def random_crop_arr_with_point(pil_image, image_size, crop_x, crop_y):
    arr = np.array(pil_image)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]

def load_vein_data(
    *, data_dir, batch_size, image_size, class_cond=None, deterministic=False
):
    if not data_dir:
        raise ValueError("unspecified data directory")
    vein_files = _list_image_files_recursively(os.path.join(data_dir, 'vein'))
    cond_files = _list_image_files_recursively(os.path.join(data_dir, 'cond'))

    dataset = VeinDataset(
        image_size,
        vein_files,
        cond_files,
        classes=class_cond,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def load_cond_data(image_size, cond_dir, batch_size, class_cond=None):
    if not cond_dir:
        raise ValueError("unspecified data directory")
    cond_files = _list_image_files_recursively(cond_dir)

    dataset = CondDataset(
        image_size,
        cond_files,
        classes=class_cond,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
    )

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=False
    )
    while True:
        yield from loader
