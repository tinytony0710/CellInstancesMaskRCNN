#  @title # model
import numpy as np
import torch
import torchvision
import torchvision.models as models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import roi_heads
from torchvision.ops.focal_loss import sigmoid_focal_loss
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import cv2

from utils.coco_utils import encode_mask


class MaskRCNN:
    def __init__(self, class_num, device, version='v1',
                 instance_num=100, lr=0.1,
                 scheduler_step=5, scheduler_rate=0.1,
                 freeze=3, enable_facol_loss=False):
        print('__init__')

        self.device = device
        if version == 'v2':
            model = models.detection.maskrcnn_resnet50_fpn_v2(
                weights='COCO_V1',
                trainable_backbone_layers=freeze
            )
        else:
            model = models.detection.maskrcnn_resnet50_fpn(
                weights='COCO_V1',
                trainable_backbone_layers=freeze
            )

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features,
            class_num
        )

        # 獲取 Mask 預測器的輸入特徵數
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        # 用一個新的 Mask 預測器替換預訓練的頭部
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask,
            hidden_layer,
            class_num
        )
        model.roi_heads.detections_per_img = instance_num

        self.model = model.to(self.device)

        # try use sigmoid_focal_loss
        if enable_facol_loss:
            roi_heads.maskrcnn_loss = sigmoid_focal_loss
            print('use sigmoid_focal_loss')

    def parameters(self):
        return self.model.parameters()

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

    def train(self, dataloader, optimizer):
        print('train')

        self.model.train()

        total_loss = 0
        total = 0

        batch_num = len(dataloader)
        for images, targets in tqdm(dataloader, total=batch_num,
                                    desc='Training', unit='batch',
                                    position=0, leave=True):
            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()}
                                                for t in targets]

            # Forward pass
            torch.cuda.synchronize(self.device) # 等待 GPU 結束
            loss_dict = self.model(images, targets)

            # Compute the loss
            losses = sum(loss for loss in loss_dict.values()
                                if torch.isfinite(loss))
            total_loss += losses.item() * len(images)
            total += len(images)

            # Zero the gradients before the backward pass
            optimizer.zero_grad()

            # Backward pass and optimization
            losses.backward()
            optimizer.step()

            # free GPU memory
            del images, targets, loss_dict
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return total_loss / total

    @torch.no_grad()
    def validate(self, dataloader):
        print('validate')

        self.model.eval()

        dataset_keys = ['id', 'image_id', 'category_id', 'bbox',
                        'area', 'iscrowd', 'segmentation']
        result_keys = ['image_id', 'category_id', 'bbox',
                       'score', 'segmentation']
        coco_data = {
            'images': [],
            'annotations': [],
            'categories': [
                {'id': 1, 'name': 'cell_type_1'},
                {'id': 2, 'name': 'cell_type_2'},
                {'id': 3, 'name': 'cell_type_3'},
                {'id': 4, 'name': 'cell_type_4'}
            ]
        }
        coco_results = []

        batch_num = len(dataloader)
        for images, targets in tqdm(dataloader, total=batch_num,
                                    desc='Validating', unit='batch',
                                    position=0, leave=True):
            images = list(image.to(self.device) for image in images)

            # torch.cuda.synchronize(self.device) # 等待 GPU 結束
            outputs = self.model(images)

            # 移到 CPU 處理
            targets = [{k: v.cpu() for k, v in t.items()} for t in targets]
            outputs = [{k: v.cpu() for k, v in t.items()} for t in outputs]

            base_index = len(coco_data['images'])
            for output, target in zip(outputs, targets):
                image_id = target['image_id'].item()
                coco_data['images'].append({
                    'id': image_id,
                    'file_name': 'image.tif',
                    'height': len(images[len(coco_data['images'])-base_index]),
                    'width': len(images[len(coco_data['images'])-base_index][0])
                })

                # target to coco_data
                """targets
                {
                    'image_id': tensor(1),
                    'boxes': tensor(N, 4),
                    'labels': tensor(N),
                    'masks': tensor(N, H, W),
                    'area': tensor(N),
                    'iscrowd': tensor(N),
                }
                """
                start = len(coco_data['annotations'])
                end = start + len(target['labels'])

                boxes = target['boxes']
                x = boxes[:, 0]
                y = boxes[:, 1]
                w = boxes[:, 2] - boxes[:, 0]
                h = boxes[:, 3] - boxes[:, 1]
                bbox = torch.stack((x, y, w, h), dim=1)

                segmentation = [encode_mask(mask) for mask in target['masks']]

                coco_data['annotations'].extend([dict(zip(dataset_keys, value))
                    for value in zip(
                        range(start, end),
                        [image_id] * len(target['labels']),
                        target['labels'].tolist(),
                        bbox.tolist(),
                        target['area'].tolist(),
                        target['iscrowd'].tolist(),
                        segmentation
                )])

                # output to coco_result
                """outputs
                {
                    'boxes':[[]]
                    'labels':[]
                    'masks':tensor(N, 1, H, W) <-- !!!!
                    'scores':[]
                }
                """
                boxes = output['boxes']
                x = boxes[:, 0]
                y = boxes[:, 1]
                w = boxes[:, 2] - boxes[:, 0]
                h = boxes[:, 3] - boxes[:, 1]
                bbox = torch.stack((x, y, w, h), dim=1)

                # 'masks':tensor(N, 1, H, W) <-- !!!!
                # segmentation = [encode_mask(mask[0] > 0.5)
                #                     for mask in output['masks']]
                segmentation = []
                for mask in output['masks']:
                    mask = mask[0] > 0.5
                    if mask.sum() > 0.12 * mask.numel():
                        continue
                    segmentation.append(encode_mask(mask))
                
                # if len(output['masks']) > 0:
                #     mask = np.array(output['masks'][0][0].tolist())
                #     cv2.imwrite('tmp.png', mask)

                coco_results.extend([dict(zip(result_keys, value))
                    for value in zip(
                        [image_id] * len(output['labels']),
                        output['labels'].tolist(),
                        bbox.tolist(),
                        output['scores'].tolist(),
                        segmentation
                )])

            # free GPU memory
            del images, targets, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # print(coco_results)
        # print(coco_data)

        # for detection(bbox):
        #  {image_id, category_id, score, bbox}
        # for segmentation(segm):
        #  {image_id, category_id, score, segmentation}
        coco_gt = COCO()
        coco_gt.dataset = coco_data
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(coco_results)
        coco_eval = COCOeval(coco_gt, coco_dt, 'segm')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        return coco_eval

    @torch.no_grad()
    def test(self, dataloader, image_name_to_ids):
        print('test')

        self.model.eval()

        result_keys = ['image_id', 'category_id', 'bbox',
                       'score', 'segmentation']
        coco_results = []

        batch_num = len(dataloader)
        for images, targets in tqdm(dataloader, total=batch_num,
                                    desc='Testing', unit='batch',
                                    position=0, leave=True):
            images = list(image.to(self.device) for image in images)
            outputs = self.model(images)
            outputs = [{k: v.cpu() for k, v in t.items()} for t in outputs]

            for output, target in zip(outputs, targets):
                image_id = image_name_to_ids[target]

                # output to coco_result
                """outputs
                {
                    'boxes':[[]]
                    'labels':[]
                    'masks':tensor(N, 1, H, W) <-- !!!!
                    'scores':[]
                }
                """
                boxes = output['boxes']
                x = boxes[:, 0]
                y = boxes[:, 1]
                w = boxes[:, 2] - boxes[:, 0]
                h = boxes[:, 3] - boxes[:, 1]
                bbox = torch.stack((x, y, w, h), dim=1)

                # 'masks':tensor(N, 1, H, W) <-- !!!!
                segmentation = []
                for mask in output['masks']:
                    mask = mask[0] > 0.5
                    if mask.sum() > 0.12 * mask.numel():
                        continue
                    segmentation.append(encode_mask(mask))

                # if len(output['masks']) > 0:
                #     mask = np.array(output['masks'][0][0].tolist())
                #     cv2.imwrite('tmp.png', mask)

                coco_results.extend([dict(zip(result_keys, value))
                    for value in zip(
                        [image_id] * len(output['labels']),
                        output['labels'].tolist(),
                        bbox.tolist(),
                        output['scores'].tolist(),
                        segmentation
                )])

        return coco_results
