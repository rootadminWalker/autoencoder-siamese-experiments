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

import torch
from torch.utils.data import DataLoader
from random import choice


class SiameseMNISTLoader(DataLoader):
    def __init__(self, mnist_dataset, batch_size, true_label=1, train_valid_split=1.):
        super(SiameseMNISTLoader, self).__init__(None, batch_size=batch_size, num_workers=4, pin_memory=True)
        self.mnist_dataset = mnist_dataset
        self.train_valid_split = train_valid_split
        self.true_label = true_label

        self.full_length = len(self.mnist_dataset) * self.batch_size
        self.train_len = int(self.full_length * self.train_valid_split)
        self.valid_len = int(1 - self.full_length * self.train_valid_split)

    def __len__(self, dataset=None):
        return self.full_length

    def __getitem__(self, idx):
        selected_im = self.mnist_dataset[idx][0]
        selected_label = self.mnist_dataset[idx][1]

        true_label_idxes = [i for i in range(len(self.mnist_dataset)) if
                            self.mnist_dataset.targets[i] == selected_label]
        false_label_idxes = [i for i in range(len(self.mnist_dataset)) if
                             self.mnist_dataset.targets[i] != selected_label]

        pair_images = [[], []]
        pair_labels = []

        for b in range(self.batch_size // 2):
            # Positive pair
            true_pair_idx = choice(true_label_idxes)
            true_image_b, image_b_label = self.mnist_dataset[true_pair_idx]
            pair_images[0].append(selected_im)
            pair_images[1].append(true_image_b)
            pair_labels.append(self.true_label)

            # Negative pair
            false_pair_idx = choice(false_label_idxes)
            false_image_b, image_b_label = self.mnist_dataset[false_pair_idx]
            pair_images[0].append(selected_im)
            pair_images[1].append(false_image_b)
            pair_labels.append(0)

        pair_images[0] = torch.stack(pair_images[0])
        pair_images[1] = torch.stack(pair_images[1])
        pair_images = torch.stack(pair_images)
        return pair_images, torch.tensor(pair_labels)

    def train(self):
        for idx in range(int(len(self.mnist_dataset) * self.train_valid_split)):
            yield self[idx]

    def validate(self):
        for idx in range(int(len(self.mnist_dataset) * self.train_valid_split), len(self)):
            yield self[idx]

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]
