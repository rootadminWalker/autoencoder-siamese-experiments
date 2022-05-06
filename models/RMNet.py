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
from typing import List

import torch
from torch import nn


class BasicRMNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate):
        super(BasicRMNetBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout_rate = dropout_rate

        self.headconv1x1 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=(1, 1),
            padding='same',
        )
        self.insideconv1x1 = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=(1, 1),
            padding='same',
        )
        self.bn = nn.BatchNorm2d(num_features=self.out_channels)
        self.elu = nn.ELU()

        self.conv3x3dw = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=(3, 3),
            padding='same',
            groups=self.out_channels
        )

        self.do = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        # Stage 1
        out = self.headconv1x1(x)
        out = self.bn(out)
        out = self.elu(out)
        # Stage 2
        out = self.conv3x3dw(out)
        out = self.bn(out)
        out = self.elu(out)
        # Final stage
        out = self.insideconv1x1(out)
        out = self.bn(out)
        out = self.do(out)

        out += x
        out = self.elu(out)
        print(f'{self._get_name()}: {out.shape}')
        return out


class SpatialReductionRMNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate):
        super(SpatialReductionRMNetBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout_rate = dropout_rate

        self.headconv1x1 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=(1, 1),
            stride=2,
            # padding='same'
        )
        self.insideconv1x1 = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=(1, 1),
            stride=2,
            # padding='same'
        )
        self.identityconv1x1 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=(1, 1),
            stride=2,
        )
        self.bn = nn.BatchNorm2d(num_features=self.out_channels)
        self.elu = nn.ELU()

        self.conv3x3dw = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=(3, 3),
            groups=self.out_channels,
            stride=1,
            padding='same'
        )

        self.do = nn.Dropout(p=dropout_rate)
        self.max_pool = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        # Main Stage 1
        main_out = self.headconv1x1(x)
        main_out = self.bn(main_out)
        main_out = self.elu(main_out)
        # Main Stage 2
        main_out = self.conv3x3dw(main_out)
        main_out = self.bn(main_out)
        main_out = self.elu(main_out)
        # Main Final stage
        main_out = self.insideconv1x1(main_out)
        main_out = self.bn(main_out)
        main_out = self.do(main_out)

        # Identity Stage
        identity_out = self.max_pool(x)
        # print(identity_out.shape)
        identity_out = self.identityconv1x1(identity_out)
        identity_out = self.bn(identity_out)

        out = main_out + identity_out
        out = self.elu(out)
        print(f'{self._get_name()}: {out.shape}')
        return out


class RMNetBackbone(nn.Module):
    def __init__(self, input_shape: tuple, block_order: list, channels_order: list, dropout_rate):
        super(RMNetBackbone, self).__init__()
        self.input_shape = input_shape
        self.block_order = block_order
        self.channels_order = channels_order
        self.dropout_rate = dropout_rate

        self.start_block = nn.Conv2d(
            in_channels=self.input_shape[1],
            out_channels=self.channels_order[0],
            kernel_size=(3, 3),
            stride=1,
            padding='same'
        )
        self.model = nn.Sequential(self.start_block)

        for idx in range(len(self.block_order)):
            for j in range(self.block_order[idx]):
                current_times = idx + 1
                current_block_num = j + 1
                current_channels = self.channels_order[idx]
                last_channels = current_channels
                if j == 0 and idx != 0:
                    last_channels = self.channels_order[idx - 1]

                if current_times % 2 != 0:
                    self.model.add_module(f'basicBlock-{current_times}-{current_block_num}', BasicRMNetBlock(
                        in_channels=last_channels,
                        out_channels=current_channels,
                        dropout_rate=dropout_rate
                    ))
                else:
                    self.model.add_module(f'spacialReductionBlock-{current_times}-{current_block_num}',
                                          SpatialReductionRMNetBlock(
                                              in_channels=last_channels,
                                              out_channels=current_channels,
                                              dropout_rate=dropout_rate
                                          ))

    def forward(self, x):
        return self.model(x)


class RMNetLinear(nn.Module):
    def __init__(self, in_features, neurons_order: List[int], activations_order: List[str]):
        super(RMNetLinear, self).__init__()
        self.in_features = in_features

        assert len(neurons_order) == len(activations_order), \
            "If you don't want act_name for a specific layer, make it to None at the corresponding location"
        self.neurons_order = neurons_order
        self.activation_order = activations_order

        self.linear = nn.Sequential(nn.Linear(
            in_features=self.in_features,
            out_features=self.neurons_order[0]
        ))
        for idx in range(1, len(self.neurons_order)):
            features = self.neurons_order[idx]
            last_features = self.neurons_order[idx - 1]
            act_name = activations_order[idx]
            current_layer = idx + 1
            self.linear.add_module(f'Linear-{current_layer}', nn.Linear(
                in_features=last_features,
                out_features=features
            ))
            if act_name is not None:
                try:
                    activation = getattr(nn, act_name)
                    self.linear.add_module(f"{act_name}-{current_layer}", activation())
                except AttributeError:
                    raise ValueError(
                        f"Activation {act_name} does not exist in torch.nn, please correct the name (custom activation is not supported yet)")

    def forward(self, x):
        assert len(x.shape) == 2, 'Must be flattened'
        return self.linear(x)


if __name__ == '__main__':
    input_shape = (1, 3, 256, 128)
    backbone = RMNetBackbone(
        input_shape=input_shape,
        block_order=[4, 1, 8, 1, 10, 1, 11],
        channels_order=[32, 64, 64, 128, 128, 256, 256],
        dropout_rate=.3
    )
    linear = RMNetLinear(
        in_features=256 * 4 * 2,
        neurons_order=[256, 512, 1024, 512, 256, 32],
        activations_order=['ReLU'] * 6
    )
    # summary(model, input_shape=input_shape)
    random_data = torch.rand(input_shape)
    random_data = backbone(random_data)
    print(random_data.shape)
    random_data = nn.Flatten()(random_data)
    print(random_data.shape)
    print(linear)
    print(linear(random_data).shape)
