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
import torch.nn.functional as F
from torch import nn


class ConvSiamese(nn.Module):
    def __init__(self, embedding_dim, conv_blocks, filters=64):
        super(ConvSiamese, self).__init__()
        blocks = []
        for _ in range(conv_blocks - 1):
            blocks.append(self.__conv_block(filters))
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
            *blocks,
            nn.Flatten(),
            nn.Linear(in_features=3136, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=embedding_dim),
        )

    @staticmethod
    def __conv_block(filters):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=filters,
                out_channels=filters,
                kernel_size=(2, 2),
                padding='same',
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(0.3),
        )

    def forward(self, x1, x2):
        embedding1 = self.model(x1)
        embedding2 = self.model(x2)
        dist = F.pairwise_distance(embedding1, embedding2)
        return dist


class OldConvSiamese(nn.Module):
    def __init__(self, embedding_dim):
        super(OldConvSiamese, self).__init__()
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
