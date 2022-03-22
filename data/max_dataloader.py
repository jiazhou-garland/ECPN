import random
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2


class CustomImageDataset(Dataset):
    def __init__(self, DataPath, val=None):
        self.DataPath = DataPath
        self.dataList = []
        self.length =self._readTXT(self.DataPath)
        self.random_indices = np.random.permutation(self.length)
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        image_path, label_path = self.dataList[idx]
        images = cv2.imread(image_path)
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)# cv2.imread read the pic with BGR form and uint8 dataform
        images = images / 255.0 # normalize the image from [0,255] to [0,1]

        label = cv2.imread(label_path)

        label = label[:, :, 1]
        label[np.where(label > 0)] = 1 # edge label -> 1
        images = np.swapaxes(np.swapaxes(images, 0, 2), 1, 2) # H,W,C -> C,H,W
        label = np.expand_dims(label, 0)  # H,W -> 1,H,W

        if self.val == True:
            return images.astype(np.float64), label.astype(np.float64), label_path
        else:
            return images.astype(np.float64), label.astype(np.float64)

    def _readTXT(self, txtPath):
        with open(txtPath, 'r') as f:
            for line in f.readlines():
                image_path, label_path = line.strip().split('\t')
                self.dataList.append((image_path,label_path))
        random.shuffle(self.dataList)
        return len(self.dataList)


def visualize (image, label):

    plt.subplot(1, 2, 1)
    plt.title('image')
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('label')
    plt.imshow(label, cmap='gray')
    plt.axis('off')

    plt.show()

if __name__ == '__main__':
    train_path = r'max_train.txt'
    datasets = CustomImageDataset(train_path)
    feeder = DataLoader(datasets, batch_size=1, shuffle=True)

    image, label = next(iter(feeder))
    image = image.permute(0,2,3,1).squeeze(axis=0) # B,C,H,W -> B,H,W,C -> H,W,C
    label = label.squeeze(axis=0).squeeze(axis=0) # B,C,H,W -> H,W

    visualize(image, label)
