import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms.v2 as T
import torchvision.transforms.functional as F
from tqdm import tqdm
from time import time
from argparse import ArgumentParser
import matplotlib.pyplot as plt

from model import MaskRCNN
from datasets import TrainDataset, TestDataset
from utils.data_io import load_json, save_json


class ToTensor:
    def __call__(self, image, target=None):
        # return image, target
        image = F.to_tensor(image)
        if target is None:
            return image
        else:
            return image, target

# 定義影像轉換 (資料增強等)
def get_transform(train):
    transforms = []
    # 將 PIL Image 或 Tensor 轉換為 Tensor
    transforms.append(ToTensor())

    return T.Compose(transforms)


if __name__ == '__main__':
    # 避免 GPU 記憶體碎片化，導致記憶體空間不足
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    # parse arguments
    parser = ArgumentParser()
    parser.add_argument(
        'directory',
        help='The directory where you save all the data. '
        + 'Should include train, val, test.'
    )
    parser.add_argument('--model-version', type=str, default='v1')
    parser.add_argument('--freeze', type=int, default=3)
    parser.add_argument('--instance-num', type=int, default=100)
    parser.add_argument('--facol-loss', action='store_true')
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--scheduler-step', type=int, default=5)
    parser.add_argument('--scheduler-rate', type=float, default=0.31622776601)
    parser.add_argument('--epoch', type=int, default=16)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # data
    data_root = args.directory
    train_data_dir = os.path.join(data_root, 'train')
    test_data_dir = os.path.join(data_root, 'test_release')
    test_json_file = os.path.join(data_root, 'test_image_name_to_ids.json')

    # model
    class_num = 4 + 1 # 4 種細胞 + 背景
    model_version = args.model_version
    freeze = args.freeze
    instance_num = args.instance_num
    enable_facol_loss = args.facol_loss

    # hyperparameter
    lr = args.lr
    scheduler_step = args.scheduler_step
    scheduler_rate = args.scheduler_rate

    # others
    epoch_num = args.epoch
    batch_size = args.batch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: ', device)

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)             # cpu
    torch.cuda.manual_seed(seed)        # current gpu
    torch.cuda.manual_seed_all(seed)    # all gpu


    # 訓練和測試資料集
    train_dataset = TrainDataset(train_data_dir, get_transform(train=True))
    test_dataset = TestDataset(test_data_dir, get_transform(train=False))

    # 隨機分割出訓練和驗證資料集
    data_size = len(train_dataset)
    train_size = int(data_size * 0.95)
    valid_size = data_size - train_size
    train_dataset, valid_dataset = random_split(
        train_dataset,
        (train_size, valid_size)
    )

    # 使用 collate_fn 處理不同大小的圖
    def collate_fn(batch):
        return tuple(zip(*batch))

    train_data_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, collate_fn=collate_fn
    )
    valid_data_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, collate_fn=collate_fn
    )

    test_data_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=2, collate_fn=collate_fn
    )


    model = MaskRCNN(class_num, device, version=model_version,
                     instance_num=instance_num, freeze=3)
    # Your model size (trainable parameters) should less than 200M.
    print('parameters: ', sum(p.numel() for p in model.parameters()
                                            if p.requires_grad))

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params,
                                lr=lr, momentum=0.9,
                                weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=scheduler_step,
                                                gamma=scheduler_rate)

    num_epochs = epoch_num
    losses = []
    mAPs = []
    mAP50s = []
    best_mAP = 0
    best_mAP50 = 0
    train_start_time = time()
    for epoch in range(num_epochs):
        print(f'Epoch #{epoch+1}')
        print(f'current lr: {scheduler.get_last_lr()}')

        loss = model.train(train_data_loader, optimizer)
        print(f'loss: {loss}')
        scheduler.step()

        coco_eval = model.validate(valid_data_loader)
        print(f'mAP: {round(coco_eval.stats[0], 4)}')
        print(f'mAP50: {round(coco_eval.stats[1], 4)}')

        if coco_eval.stats[0] > best_mAP:
            best_mAP = coco_eval.stats[0]
            model.save_model('best_mAP_model')
            print(f'best_mAP: {best_mAP}')
        if coco_eval.stats[1] > best_mAP50:
            best_mAP50 = coco_eval.stats[1]
            model.save_model('best_mAP50_model')
            print(f'best_mAP50: {best_mAP50}')
        
        losses.append(loss)
        mAPs.append(coco_eval.stats[0])
        mAP50s.append(coco_eval.stats[1])
        
    train_end_time = time()
    print(f'{num_epochs} 個 epoch，共計 {train_end_time - train_start_time} 秒')

    plt.plot(losses, 'r-', label='loss')
    plt.plot(mAPs, 'g-', label='mAP')
    plt.plot(mAP50s, 'g--', label='mAP50')
    plt.legend()
    plt.savefig('runtime_stat.png')
    plt.show()
    
    # test_dataset
    model.load_model('best_mAP_model')
    json_data = load_json(test_json_file)
    image_name_to_ids = {item['file_name']: item['id'] for item in json_data}
    result = model.test(test_data_loader, image_name_to_ids)
    save_json('test-results.json', result)
