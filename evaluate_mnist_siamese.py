from random import choice

import cv2 as cv
import numpy as np
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class ConvSiamese(nn.Module):
    def __init__(self, embedding_dim):
        super(ConvSiamese, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=(2, 2),
                padding='same',
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(0.3),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(2, 2),
                padding='same',
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(0.3),
            nn.Flatten(),
            nn.Linear(in_features=3136, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=embedding_dim),
        )

    def forward(self, x1, x2):
        embedding1 = self.model(x1)
        embedding2 = self.model(x2)
        dist = F.pairwise_distance(embedding1, embedding2)
        return dist


class SiameseMNISTLoader(DataLoader):
    def __init__(self, mnist_dataset, batch_size):
        super(SiameseMNISTLoader, self).__init__(None, batch_size=batch_size, num_workers=4, pin_memory=True)
        self.mnist_dataset = mnist_dataset
        self.full_length = len(self.mnist_dataset) * self.batch_size

    def __len__(self):
        return len(self.mnist_dataset)

    def __iter__(self):
        for idx in range(len(self)):
            selected_im = self.mnist_dataset[idx][0]
            selected_label = self.mnist_dataset[idx][1]

            true_label_idxes = [i for i in range(len(self.mnist_dataset)) if
                                self.mnist_dataset.targets[i] == selected_label]
            false_label_idxes = [i for i in range(len(self.mnist_dataset)) if
                                 self.mnist_dataset.targets[i] != selected_label]

            pair_images, pair_labels = [], []

            for b in range(self.batch_size // 2):
                # Positive pair
                true_pair_idx = choice(true_label_idxes)
                true_image_b = self.mnist_dataset[true_pair_idx][0]
                true_image_pair = [selected_im, true_image_b]
                pair_images.append(torch.stack(true_image_pair))
                pair_labels.append(1)

                # Negative pair
                false_pair_idx = choice(false_label_idxes)
                false_image_b = self.mnist_dataset[false_pair_idx][0]
                false_image_pair = [selected_im, false_image_b]
                pair_images.append(torch.stack(false_image_pair))
                pair_labels.append(0)

            pair_images = torch.stack(pair_images).reshape((2, self.batch_size, 1, 28, 28))
            yield pair_images, torch.tensor(pair_labels)


if __name__ == '__main__':
    device = 'cuda:0'
    model = ConvSiamese(embedding_dim=48)
    model.load_state_dict(torch.load('outputs/mnist_siamese_outputs/embedding_dim_48_ep10/siamese.pth'))
    model.eval()
    model.to(device)

    test_dataset = torchvision.datasets.MNIST(
        root='/tmp',
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )
    test_dataloader = SiameseMNISTLoader(test_dataset, batch_size=32)

    for batch_images, batch_labels in test_dataloader:
        batch_images = batch_images.reshape((32, 2, 1, 28, 28))
        for pair_image, label in zip(batch_images, batch_labels):
            imageA, imageB = pair_image
            imageA = torch.stack([imageA]).to(device)
            imageB = torch.stack([imageB]).to(device)
            dist = model(imageA, imageB)

            imageA = np.array(imageA.cpu()).reshape((28, 28, 1)) * 255
            imageA = imageA.astype('uint8')
            imageB = np.array(imageB.cpu()).reshape((28, 28, 1)) * 255
            imageB = imageB.astype('uint8')
            hstacked = np.hstack([imageA, imageB])
            print(f'Estimated Dist: {dist}, True label: {label}')

            cv.imshow('frame', hstacked)
            cv.waitKey(0)
