import random, torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2
import albumentations as A
from PIL import Image

class CustomImageDataset(Dataset):
    def __init__(self, DataPath, single_polarized = None, val = False):
        self.DataPath = DataPath
        self.dataList = []
        self.single_polarized = single_polarized
        self.length =self._readTXT(self.DataPath)
        self.random_indices = np.random.permutation(self.length)
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.val = val

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        image_path1, image_path2, image_path3, image_path4, \
        image_path5, image_path6, image_path7, label_path = self.dataList[idx]
        n = 7

        images = np.empty((512, 512, 0), dtype=np.uint8)

        for i in range(n):
            image = cv2.imread(eval("image_path"+str(i+1)))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)# cv2.imread read the pic with BGR form and uint8 dataform
            # image = self.normalize(image, self.mean, self.std)
            image = image / 255.0 # normalize the image from [0,255] to [0,1]
            images = np.concatenate((images, image), axis=2)

        label = cv2.imread(label_path)

        label = label[:, :, 1] # # H,W,3 -> H,W,1
        label[np.where(label > 0)] = 1 # edge label -> 1
        images = np.swapaxes(np.swapaxes(images, 0, 2), 1, 2) # H,W,C -> C,H,W
        label = np.expand_dims(label, 0)  # H,W -> 1,H,W

        if self.val == True:
            return images, label, label_path
        else :
            return  images, label

    def _readTXT(self, txtPath):
        with open(txtPath, 'r') as f:
	    for line in f.readlines():
		    image_path1, image_path2, image_path3, image_path4, \
		    image_path5, image_path6, image_path7, image_path8, label_path = line.strip().split('\t')
		    self.dataList.append((image_path2, image_path3, image_path4, image_path5,
			                  image_path6, image_path7, image_path8, label_path))
        random.shuffle(self.dataList)
        return len(self.dataList)

    def normalize(self, img, mean, std, max_pixel_value=255.0):
        mean = np.array(mean, dtype=np.float32)
        mean *= max_pixel_value

        std = np.array(std, dtype=np.float32)
        std *= max_pixel_value

        denominator = np.reciprocal(std, dtype=np.float32)

        img = img.astype(np.float32)
        img -= mean
        img *= denominator
        return img

def visualize (images, label):

    for i in range(7):
        plt.subplot(3, 3, (i+1)),
        plt.title('image'+str(i))
        plt.imshow(images[:,: , 3*i:3*(i+1)])
        plt.axis('off')

    plt.subplot(3, 3, 9)
    plt.title('label')
    plt.imshow(label, cmap='gray')
    plt.axis('off')

    plt.show()

def visualize_label (gt, inverted_gt, D_, D):

    plt.subplot(1, 4, 1)
    plt.title('gt')
    plt.imshow(gt, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.title('inverted_gt')
    plt.imshow(inverted_gt, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.title('D_')
    plt.imshow(D_, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.title('D')
    plt.imshow(D, cmap='gray')
    plt.axis('off')

    plt.show()

def distancemap_penalized_display(targets, maskSize=5):

    eps = 1e-9
    h, w, c = targets.size()
    tmp = np.zeros((h, w), dtype=np.uint8)

    t = targets.squeeze(axis=2).cpu().data.numpy() #hxwx1-> hxw

    # inverter of the target
    tmp[np.where(t == 1)] = 0
    tmp[np.where(t == 0)] = 1

    distance_map = cv2.distanceTransform(tmp, cv2.DIST_L2, maskSize=maskSize)
    distance_weights = 1 / (distance_map + eps)
    distance_weights[tmp == 0] = 1

    distance_weights = distance_weights[:, :, np.newaxis]
    return tmp, distance_map, distance_weights

if __name__ == '__main__':
    train_path = r'train.txt'
    datasets = CustomImageDataset(train_path)
    feeder = DataLoader(datasets, batch_size=1, shuffle=True)

    image, label = next(iter(feeder))
    image = image.squeeze(axis=0).transpose(0,2).transpose(0,1)
    label = label.squeeze(axis=0).transpose(0,2).transpose(0,1)

    visualize(image, label)

    inverted_gt, D_, D = distancemap_penalized_display(label, maskSize=5)
    visualize_label(label, inverted_gt, D_, D)
