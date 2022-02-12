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
from .SimpleConvSiamese import ConvSiamese


class ConvTriplet(ConvSiamese):
    def __init__(self, embedding_dim, conv_blocks, filters=64):
        super(ConvTriplet, self).__init__(embedding_dim, conv_blocks, filters)

    def forward(self, anchor, true_positive, false_positive):
        embedding_anchor = self.model(anchor)
        embedding_true = self.model(true_positive)
        embedding_false = self.model(false_positive)
        anchor_positive_dist = F.pairwise_distance(embedding_anchor, embedding_true)
        anchor_negative_dist = F.pairwise_distance(embedding_anchor, embedding_false)

        return anchor_positive_dist, anchor_negative_dist
