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
from torch import nn


class MarginMSELoss(nn.Module):
    def __init__(self, margin):
        super(MarginMSELoss, self).__init__()
        self.margin = margin
        self.mse = nn.MSELoss()

    def forward(self, dist, label):
        preprocessed_label = (self.margin - label).abs().type(torch.float)
        loss = self.mse(dist, preprocessed_label)
        return loss


class ContrastiveLoss(nn.Module):
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def contrastive_loss(self, y, D):
        return torch.mean(y * D.pow(2) + (1 - y) * torch.pow(torch.clamp(self.margin - D, min=0.0), 2))

    def forward(self, dist, label):
        loss = self.contrastive_loss(label, dist)
        return loss


class TripletLoss(nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def triplet_loss(self, anchor_positive_dist, anchor_negative_dist):
        return torch.mean(torch.clamp(anchor_positive_dist - anchor_negative_dist + self.margin, min=0.0))

    def forward(self, anchor_positive_dist, anchor_negative_dist):
        return self.triplet_loss(anchor_positive_dist, anchor_negative_dist)
