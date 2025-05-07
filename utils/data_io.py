from os import listdir, path
import json
import csv
import numpy as np
from PIL import Image
import skimage.io as sio


def load_tif_image(path, gray=False):
    image = load_image(path, gray)
    return image

def load_tif_mask(path):
    mask = sio.imread(path)
    return mask

def load_image(path, gray=False):
    # print(path)
    image = Image.open(path)
    if gray:
        image = image.convert('L')
    else:
        image = image.convert('RGB')
    # image = image.resize((224, 224))
    image = np.array(image)
    # if data_type == 'int':
    #     image = np.array(image, dtype=np.uint8)
    # elif data_type == 'float':
    #     image = np.array(image, dtype=np.float32) / 255.0
    # print(image.shape)
    return image

def load_images_in_folder(folder_path):
    print(folder_path)
    images = []
    for filename in sorted(listdir(folder_path), key=int):
        image = load_image(path.join(folder_path, filename))
        images.append(image)
    images = np.array(images)
    return images

def load_dataset(path):
    print(path)
    images = []
    for folder_name in sorted(listdir(path), key=int):
        folder_path = path.join(path, folder_name)
        folder_images = load_images_in_folder(folder_path)
        images.append(folder_images)
    return images

def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f)

def save_csv(path, data, header=None):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        if header: writer.writerow(header)
        for row in data:
            writer.writerow(row)
