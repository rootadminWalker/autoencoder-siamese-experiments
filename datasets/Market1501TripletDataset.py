"""
MIT License

Copyright (c) 2021 Thomas Leong

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

import os
from copy import copy
from random import choice, randint
from typing import Optional

import cv2 as cv
import numpy as np
import torch
import torch.utils.data
from torch.utils.data import Dataset


class Market1501Dataset(Dataset):
    TRAIN_IMAGES_AMOUNT = 0

    def __init__(
            self,
            dataset_path: str,
            device: str,
            batch_size: int,
            transforms
    ):
        self.dataset_path = dataset_path
        self.device = device
        self.batch_size = batch_size
        self.transforms = transforms

        self.train_path = os.path.join(self.dataset_path, 'bounding_box_train')
        Market1501Dataset.TRAIN_IMAGES_AMOUNT = len(self.__list_dataset_images(self.train_path))
        self.full_length = 0

        self.dataset = {'labels_to_path_idx': {}, 'image_paths': []}
        self.__load_dataset(self.train_path)

    @staticmethod
    def __list_dataset_images(train_path):
        train_images = os.listdir(train_path)
        train_images.remove('Thumbs.db')
        return train_images

    def __load_dataset(self, train_path):
        train_images = self.__list_dataset_images(train_path)
        for idx, image_name in enumerate(train_images):
            abs_image_path = os.path.join(self.train_path, image_name)
            individual = self.get_label_from_path(abs_image_path)

            if individual not in self.dataset['labels_to_path_idx']:
                self.dataset['labels_to_path_idx'][individual] = []

            self.dataset['labels_to_path_idx'][individual].append(idx)
            self.dataset['image_paths'].append(abs_image_path)

        self.dataset['labels_to_path_idx'] = {
            k: v for k, v in
            sorted(self.dataset['labels_to_path_idx'].items(), key=lambda item: int(item[0].split('_')[0]))
        }

    def read_image_from_device(self, path, device):
        image = cv.imread(path)
        converted_image = torch.tensor(image, device=device).permute(2, 0, 1)
        return self.transforms(converted_image)

    @staticmethod
    def get_label_from_path(image_path):
        return image_path.split('/')[-1].split('_')[0]

    def set_full_length(self, full_length):
        self.full_length = full_length

    def __len__(self):
        return self.full_length

    def __getitem__(self, index):
        if isinstance(index, slice):
            start_idx = index.start
            stop_idx = index.stop
            if start_idx is None:
                start_idx = 0
            if stop_idx is None:
                stop_idx = self.full_length - 1

            if stop_idx >= self.full_length:
                raise IndexError(f"Dataset only has {self.full_length} images")
        else:
            start_idx = index
            stop_idx = index + 1
            if start_idx >= self.full_length:
                raise IndexError(f"Dataset only has {self.full_length} images")

        return start_idx, stop_idx


class SiameseMarket1501Dataset(Market1501Dataset):
    def __init__(
            self,
            dataset_path: str,
            device: str,
            batch_size: int,
            transforms,
            image_limit: Optional[int] = None,
            pairs_per_image: Optional[int] = None
    ):
        super(SiameseMarket1501Dataset, self).__init__(dataset_path, device, batch_size, transforms)

        self.pairs_per_image = pairs_per_image
        if self.pairs_per_image is None:
            self.pairs_per_image = self.batch_size

        if image_limit is None:
            image_limit = SiameseMarket1501Dataset.TRAIN_IMAGES_AMOUNT * self.pairs_per_image

        self.set_full_length(image_limit)

    def __getitem__(self, index) -> torch.tensor:
        start_idx, stop_idx = super(SiameseMarket1501Dataset, self).__getitem__(index)

        pair_images = torch.tensor([], device=self.device)
        pair_labels = torch.tensor([0], device=self.device)

        labels_to_path_idx = self.dataset['labels_to_path_idx']
        image_paths = self.dataset['image_paths']

        for idx in range(start_idx, stop_idx):
            image1_idx = idx // self.pairs_per_image
            image1_path = image_paths[image1_idx]
            image1_label = self.get_label_from_path(image1_path)
            image1 = self.read_image_from_device(image1_path, self.device).unsqueeze(0)

            # Remove the anchors from the individual's idx list
            isPositive = randint(0, 1)
            if isPositive:
                same_label_pool = labels_to_path_idx[image1_label]
                image2_path = image_paths[choice(same_label_pool)]
            else:
                image2_path = choice(image_paths)
            image2_label = self.get_label_from_path(image2_path)
            image2 = self.read_image_from_device(image2_path, self.device).unsqueeze(0)

            pair = torch.stack((image1, image2))
            pair_images = torch.cat((pair_images, pair), dim=0)
            pair_label = torch.tensor([1 if image1_label == image2_label else 0], device=self.device)
            pair_labels = torch.stack((pair_labels, pair_label))

        return pair_labels[1:], pair_images


class TripletMarket1501Dataset(Market1501Dataset):
    def __init__(
            self,
            dataset_path: str,
            device: str,
            batch_size: int,
            transforms,
            image_limit: Optional[int] = None,
            triplets_per_anchor: Optional[int] = None
    ):
        super(TripletMarket1501Dataset, self).__init__(dataset_path, device, batch_size, transforms)

        self.triplets_per_anchor = triplets_per_anchor
        if self.triplets_per_anchor is None:
            self.triplets_per_anchor = self.batch_size

        if image_limit is None:
            image_limit = TripletMarket1501Dataset.TRAIN_IMAGES_AMOUNT * self.triplets_per_anchor

        self.set_full_length(image_limit)

    def get_dataset_histogram(self, show_original=False):
        show_triplets = not show_original
        hist = []
        for k, v in self.dataset['labels_to_path_idx_to_idx'].items():
            quantity = [str(k)] * int(len(v))
            quantity = max(quantity * show_triplets * self.triplets_per_anchor, quantity)
            hist.extend(quantity)

        hist = map(int, hist)
        return np.array(list(hist))

    def __len__(self):
        return self.full_length

    def __getitem__(self, index) -> torch.tensor:
        start_idx, stop_idx = super(TripletMarket1501Dataset, self).__getitem__(index)

        triplets = torch.tensor([], device=self.device)

        labels_to_path_idx = self.dataset['labels_to_path_idx']
        image_paths = self.dataset['image_paths']
        labels = list(labels_to_path_idx.keys())

        for idx in range(start_idx, stop_idx):
            anchor_idx = idx // self.triplets_per_anchor
            anchor_path = image_paths[anchor_idx]
            anchor_label = self.get_label_from_path(anchor_path)
            anchor = self.read_image_from_device(anchor_path, self.device)

            # Remove the anchors from the individual's idx list
            anchor_removed_pool = copy(labels_to_path_idx[anchor_label])
            del anchor_removed_pool[anchor_removed_pool.index(anchor_idx)]
            positive_idx = choice(anchor_removed_pool)
            positive_path = image_paths[positive_idx]
            positive = self.read_image_from_device(positive_path, self.device)

            # Remove the positive label from individual_ids
            positive_removed_ids = copy(labels)
            del positive_removed_ids[positive_removed_ids.index(anchor_label)]
            positive_removed_pool = labels_to_path_idx[choice(positive_removed_ids)]
            negative_idx = choice(positive_removed_pool)
            negative_path = image_paths[negative_idx]
            negative = self.read_image_from_device(negative_path, self.device)

            triplet = torch.stack((anchor, positive, negative))
            triplets = torch.cat((triplets, triplet), dim=0)

        return triplets


if __name__ == '__main__':
    dataset = TripletMarket1501Dataset("/tmp/Market-1501-v15.09.15", "cuda:0", 32, triplets_per_anchor=12)
    train = torch.utils.data.Subset(dataset, list(range(int(len(dataset) * 0.8))))
    valid = torch.utils.data.Subset(dataset, list(range(int(len(dataset) * 0.8), len(dataset))))
    print(train, valid)
