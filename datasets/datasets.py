#  @title # datasets
import os
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.data_io import load_image, load_tif_mask


class TrainDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.image_names = sorted(os.listdir(root))
        self.transforms = transforms
        print(f'{root}: 共 {len(self.image_names)} 張圖像')

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        """
        return 圖像: tensor(C, H, W), {
            'image_id': tensor(1),
            'boxes': tensor(N, 4),
            'labels': tensor(N),
            'masks': tensor(N, H, W),
            'area': tensor(N),
            'iscrowd': tensor(N),
        }
        """

        image_dir = os.path.join(self.root, self.image_names[idx])
        image_path = os.path.join(image_dir, 'image.tif')
        image = load_image(image_path)

        boxes = []
        labels = []
        instance_masks = []
        area = []
        iscrowd = []

        # 尋找該目錄下所有的 class*.tif
        mask_names = sorted(os.listdir(image_dir))
        for mask_name in mask_names:
            if not mask_name.startswith('class'): continue

            mask_path = os.path.join(image_dir, mask_name)
            mask = load_tif_mask(mask_path)
            id = int(os.path.splitext(mask_name.replace('class', ''))[0])

            start = int(mask[mask > 0].min())
            end = int(mask.max()) + 1
            for i in range(start, end):
                instance_mask = (mask == i).astype(np.uint8)
                if instance_mask.sum() == 0: continue

                pos = np.where(instance_mask)
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                boxes.append([xmin, ymin, xmax, ymax])

                labels.append(id)
                instance_masks.append(instance_mask)
                area.append(np.sum(instance_mask))
                iscrowd.append(0)

        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            instance_masks = np.array(instance_masks)
            instance_masks = torch.as_tensor(instance_masks, dtype=torch.uint8)
            area = torch.as_tensor(area, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        else:
            print('warning: no boxes')
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            instance_masks = torch.zeros((0, image.shape[1], image.shape[2]),
                                         dtype=torch.uint8)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)

        target = {}
        target['image_id'] = torch.tensor([idx])
        target['boxes'] = boxes
        target['labels'] = labels
        target['masks'] = instance_masks
        target['area'] = area
        target['iscrowd'] = iscrowd

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        
        # size = len(target['labels'])
        # print(size)
        # if size > 700:
        #     # randomly select
        #     indices = np.random.choice(size, 700, replace=False)
        #     target['labels'] = target['labels'][indices]
        #     target['masks'] = target['masks'][indices]
        #     target['boxes'] = target['boxes'][indices]
        #     target['area'] = target['area'][indices]
        #     target['iscrowd'] = target['iscrowd'][indices]

        return image, target


class TestDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.image_names = sorted(os.listdir(root))
        self.transforms = transforms
        print(f'{root}: 共 {len(self.image_names)} 張圖像')

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        """
        return 圖像: tensor(C, H, W), 圖像名稱: str
        """

        image_id = self.image_names[idx]
        image_path = os.path.join(self.root, image_id)
        image = load_image(image_path)

        if self.transforms is not None:
            image = self.transforms(image)

        return image, image_id
