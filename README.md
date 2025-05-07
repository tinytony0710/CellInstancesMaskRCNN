# CellInstancesMaskRCNN

## Introduction

A fine-tuned over both pretrained `maskrcnn_resnet50_fpn` and `maskrcnn_resnet50_fpn_v2` model for 4 types of cell instances segmentation.

## Environment

On colab: Should be able to train with batch size 1 and 700 instances per images, and inference 256 instances per images.

## Usage

```bash
python main.py <data-directory> [options]
```

**Needed:**

`<data-directory>`: The path to where you store data, the directory should be like the following:

```
--root
  |--train
  |  |--<folder1>
  |  |  |--image.tif
  |  |  |--class1.tif
  |  |  |  ...
  |--test_release
  |  |--<image1>.tif
  |  |  ...
```

**Options:**

* `--model-version`: Choose v1 (maskrcnn_resnet50_fpn) and v2 (maskrcnn_resnet50_fpn_v2). Default='v1'.

* `--freeze`: The number of layers you DON'T WANT to freeze. Default=3, integer between 1 to 6.

* `--instance-num`: Maximum number of instances to predict per image during inference. Default=100.

* `--lr`: Learning rate. Default=0.0002

* `--scheduler-step`: Step size for the learning rate scheduler. Default=5

* `--scheduler-rate`: Rate to multiply the learning rate by at each scheduler step. Default=0.31622776601

* `--epoch`: Number of training epochs. Default=16

* `--batch`: Batch size for training. Default=4

* `--seed`: Random seed for reproducibility. Default=42

**Example Command (Configuration yielding best result):**

```bash
python main.py /path/to/your/data --epoch 12 --scheduler-step 3 --batch 1 --seed 774 --predict-instance-num 256 --model-version 'v1'
```
The program will run training, validation and testing individually. The submission file for the test dataset will be saved as `test-results.json`.

## Results

Val AP@0.50-0.95: 0.35

Val AP@0.50: 0.50

Test AP@0.50-0.95: 0.4054
