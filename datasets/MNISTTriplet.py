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

from random import choice

import torch
from torch.utils.data import DataLoader


class TripletMNISTLoader(DataLoader):
    def __init__(self, mnist_dataset, device, batch_size, train_valid_split=1., image_limit=None,
                 triplets_per_anchor=None):
        super(TripletMNISTLoader, self).__init__(None, batch_size=batch_size, num_workers=4, pin_memory=True)
        self.mnist_dataset = mnist_dataset
        self.device = device
        self.train_valid_split = train_valid_split

        self.triplets_per_anchor = triplets_per_anchor
        if self.triplets_per_anchor is None:
            self.triplets_per_anchor = self.batch_size

        if image_limit is None:
            image_limit = len(self.mnist_dataset) * self.triplets_per_anchor

        self.full_length = image_limit
        self.train_len = int(self.full_length * self.train_valid_split)
        self.valid_len = int(self.full_length * (1 - self.train_valid_split))

    def __len__(self, dataset=None):
        return self.full_length

    def __get_basic_data(self, anchor_idx):
        anchor = self.mnist_dataset[anchor_idx][0]
        anchor_label = self.mnist_dataset[anchor_idx][1]

        true_label_idxes = [i for i in range(len(self.mnist_dataset)) if
                            self.mnist_dataset.targets[i] == anchor_label]
        false_label_idxes = [i for i in range(len(self.mnist_dataset)) if
                             self.mnist_dataset.targets[i] != anchor_label]

        return anchor, anchor_label, true_label_idxes, false_label_idxes

    def __generate_pair_from_basic_info(self, anchor, true_label_idxes, false_label_idxes):
        pair_image = [[], [], []]

        # Triplet pair (anchor, true positive, false_positive)
        true_pair_idx = choice(true_label_idxes)
        false_pair_idx = choice(false_label_idxes)
        true_image, _ = self.mnist_dataset[true_pair_idx]
        false_image, _ = self.mnist_dataset[false_pair_idx]
        pair_image[0].append(anchor)
        pair_image[1].append(true_image)
        pair_image[2].append(false_image)

        pair_image[0] = torch.stack(pair_image[0])
        pair_image[1] = torch.stack(pair_image[1])
        pair_image[2] = torch.stack(pair_image[2])
        pair_image = torch.stack(pair_image)
        return pair_image

    def train(self):
        pair_images = torch.tensor([], device=self.device)
        anchor, _, true_label_idxes, false_label_idxes = self.__get_basic_data(0)

        for idx in range(self.train_len):
            triplet = self.__generate_pair_from_basic_info(anchor, true_label_idxes, false_label_idxes)
            pair_images = torch.cat((pair_images, triplet), dim=1)
            if (idx + 1) % self.triplets_per_anchor == 0:
                anchor_idx = (idx + 1) // self.triplets_per_anchor
                anchor, _, true_label_idxes, false_label_idxes = self.__get_basic_data(anchor_idx)

            if len(pair_images[0]) == self.batch_size or idx == (self.train_len - 1):
                yield pair_images
                pair_images = torch.tensor([], device=self.device)

    def validate(self):
        pair_images = torch.tensor([], device=self.device)
        anchor, _, true_label_idxes, false_label_idxes = self.__get_basic_data(0)

        for idx in range(self.train_len):
            triplet = self.__generate_pair_from_basic_info(anchor, true_label_idxes, false_label_idxes)
            pair_images = torch.cat((pair_images, triplet), dim=1)
            if (idx + 1) % self.triplets_per_anchor == 0:
                anchor_idx = (idx + 1) // self.triplets_per_anchor
                anchor, _, true_label_idxes, false_label_idxes = self.__get_basic_data(anchor_idx)

            if len(pair_images[0]) == self.batch_size or idx == (self.train_len - 1):
                yield pair_images
                pair_images = torch.tensor([], device=self.device)

    def __iter__(self):
        yield self.train()
        yield self.validate()
