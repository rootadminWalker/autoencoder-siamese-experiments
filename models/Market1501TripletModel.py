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
import torch.nn.functional as F
from torch import nn


class Market1501TripletModel(nn.Module):
    MODEL_NAME = "Self-made cnn"

    def __init__(
            self,
            input_shape,
            embedding_dim,
            conv_blocks,
            conv_kernel_size,
            max_pool_kernel_size,
            dropout_rate,
            filters=64,
    ):
        super(Market1501TripletModel, self).__init__()

        blocks = []
        start_block = self.__conv_block(
            in_channels=input_shape[1],
            out_channels=filters,
            conv_kernel_size=conv_kernel_size,
            max_pool_kernel_size=max_pool_kernel_size,
            dropout_rate=dropout_rate
        )
        for _ in range(conv_blocks - 1):
            blocks.append(self.__conv_block(
                in_channels=filters,
                out_channels=filters,
                conv_kernel_size=conv_kernel_size,
                max_pool_kernel_size=max_pool_kernel_size,
                dropout_rate=dropout_rate
            ))
        self.backbone = nn.Sequential(
            start_block,
            *blocks,
            nn.Flatten(),
        )

        backbone_output_features = self.backbone(torch.rand(input_shape)).shape[-1]
        self.fully_connected = nn.Sequential(
            nn.Linear(in_features=backbone_output_features, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=embedding_dim)
        )
        self.model = nn.Sequential(self.backbone, self.fully_connected)

    @staticmethod
    def __conv_block(in_channels, out_channels, conv_kernel_size, max_pool_kernel_size, dropout_rate):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=conv_kernel_size,
                padding='same',
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=max_pool_kernel_size),
            nn.Dropout(dropout_rate),
        )

    def forward(self, anchor, true_positive, false_positive):
        embedding_anchor = self.model(anchor)
        embedding_true = self.model(true_positive)
        embedding_false = self.model(false_positive)
        anchor_positive_dist = F.pairwise_distance(embedding_anchor, embedding_true)
        anchor_negative_dist = F.pairwise_distance(embedding_anchor, embedding_false)

        return anchor_positive_dist, anchor_negative_dist


class Market1501TripletMiniVGG(nn.Module):
    INITIAL_FILTERS = 32
    MODEL_NAME = "Mini-VGG"

    def __init__(
            self,
            input_shape,
            embedding_dim,
            conv_blocks,
            conv_kernel_size=(3, 3),
            max_pool_kernel_size=(2, 2),
            dropout_rate=0.03,
            filters=64
    ):
        super(Market1501TripletMiniVGG, self).__init__()
        self.model_name = "Mini-VGG"

        start_block = self.__conv_block(
            in_channels=input_shape[1],
            out_channels=Market1501TripletMiniVGG.INITIAL_FILTERS,
            conv_kernel_size=conv_kernel_size,
        )
        self.backbone = nn.Sequential(start_block)

        assert conv_blocks % 2 == 0, "Conv blocks must be an even number in MiniVGGNet"
        for idx in range(2, conv_blocks + 1):
            current_filters_multiply = int(round((idx / 2) + 0.1, 0))
            current_filters = Market1501TripletMiniVGG.INITIAL_FILTERS * current_filters_multiply

            _, last_output_channels, _, _ = self.__get_last_shape(input_shape, self.backbone)
            self.backbone.add_module(f'Block:{idx}', self.__conv_block(
                in_channels=last_output_channels,
                out_channels=current_filters,
                conv_kernel_size=conv_kernel_size,
            ))
            if idx % 2 == 0:
                self.backbone.add_module(f'Pool:{current_filters_multiply}', self.__pool_block(
                    max_pool_kernel_size=max_pool_kernel_size,
                    dropout_rate=dropout_rate
                ))

        self.backbone.add_module('Flatten', nn.Flatten())
        backbone_output_features = self.__get_last_shape(input_shape, self.backbone)[-1]
        self.fully_connected = nn.Sequential(
            nn.Linear(in_features=backbone_output_features, out_features=512),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=512),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=128),
            nn.Linear(in_features=128, out_features=embedding_dim)
        )
        self.model = nn.Sequential(self.backbone, self.fully_connected)

    @staticmethod
    def __get_last_shape(input_shape, block):
        return block(torch.rand(input_shape)).shape

    @staticmethod
    def __pool_block(max_pool_kernel_size, dropout_rate):
        return nn.Sequential(
            nn.MaxPool2d(kernel_size=max_pool_kernel_size),
            nn.Dropout(p=dropout_rate)
        )

    @staticmethod
    def __conv_block(in_channels, out_channels, conv_kernel_size):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=conv_kernel_size,
                padding='same',
                bias=True
            ),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=out_channels)
        )

    def forward(self, anchor, positive, negative):
        anchor_emd = self.model(anchor)
        positive_emd = self.model(positive)
        negative_emd = self.model(negative)

        anchor_positive_dist = F.pairwise_distance(anchor_emd, positive_emd)
        anchor_negative_dist = F.pairwise_distance(anchor_emd, negative_emd)

        return anchor_positive_dist, anchor_negative_dist


class Market1501TripletModelEval(Market1501TripletModel):
    def __init__(
            self,
            input_shape,
            embedding_dim,
            conv_blocks,
            conv_kernel_size=(3, 3),
            max_pool_kernel_size=(2, 2),
            dropout_rate=0.03,
            filters=64
    ):
        super(Market1501TripletModelEval, self).__init__(
            input_shape=input_shape,
            embedding_dim=embedding_dim,
            conv_blocks=conv_blocks,
            conv_kernel_size=conv_kernel_size,
            max_pool_kernel_size=max_pool_kernel_size,
            dropout_rate=dropout_rate,
            filters=filters
        )

    def forward(self, image):
        return self.model(image)


class Market1501TripletMiniVGGEval(Market1501TripletMiniVGG):
    def __init__(
            self,
            input_shape,
            embedding_dim,
            conv_blocks,
            conv_kernel_size=(3, 3),
            max_pool_kernel_size=(2, 2),
            dropout_rate=0.03,
            filters=64
    ):
        super(Market1501TripletMiniVGGEval, self).__init__(
            input_shape=input_shape,
            embedding_dim=embedding_dim,
            conv_blocks=conv_blocks,
            conv_kernel_size=conv_kernel_size,
            max_pool_kernel_size=max_pool_kernel_size,
            dropout_rate=dropout_rate,
            filters=filters
        )

    def forward(self, image):
        return self.model(image)


if __name__ == '__main__':
    from torchinfo import summary

    input_shape = (1, 3, 128, 64)
    model = Market1501TripletMiniVGG(
        input_shape,
        embedding_dim=32,
        conv_blocks=2,
    )

    summary(model, (input_shape, input_shape, input_shape))
