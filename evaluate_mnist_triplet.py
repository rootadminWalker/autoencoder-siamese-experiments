import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import utils

from datasets import TripletMarket1501Dataset
from models import OldConvSiamese


class ConvSiamese(OldConvSiamese):
    def __init__(self, embedding_dim):
        super(ConvSiamese, self).__init__(embedding_dim)

    def forward(self, x1, x2):
        embedding1 = self.model(x1)
        embedding2 = self.model(x2)
        dist = F.pairwise_distance(embedding1, embedding2)
        return dist, embedding1, embedding2


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    device = 'cuda:0'
    # model = ConvSiamese(embedding_dim=48)
    # model.load_state_dict(torch.load(
    #     '/media/rootadminwalker/DATA/outputs/mnist_siamese_outputs/embedding_dim_48_ep15_loss_contrastive_margin_3/siamese.pth'))
    # model.eval()
    # summary(model, input_size=((1, 1, 28, 28), (1, 1, 28, 28)))
    # model.to(device)

    dataset = TripletMarket1501Dataset('/media/rootadminwalker/DATA/datasets/Market-1501-v15.09.15', device='cuda:0',
                                       batch_size=32,
                                       similar_identities_cfg_path='./similar_identities.json',
                                       triplets_per_anchor=12)
    train_valid_split = 0.8
    train = torch.utils.data.Subset(dataset, list(range(int(len(dataset) * train_valid_split))))
    valid = torch.utils.data.Subset(dataset, list(range(int(len(dataset) * train_valid_split), len(dataset))))

    test_dataloader = DataLoader(train, batch_size=32, num_workers=0)

    for batch_images in test_dataloader:
        print(batch_images.shape)
        for idx in range(batch_images.shape[0]):
            imageA = utils.float_to_255(batch_images[idx, 0])
            imageB = utils.float_to_255(batch_images[idx, 1])
            imageC = utils.float_to_255(batch_images[idx, 2])
            print(imageA.dtype)

            imageA = utils.torch_to_cv2(imageA)
            imageB = utils.torch_to_cv2(imageB)
            imageC = utils.torch_to_cv2(imageC)

            hstacked = np.hstack([imageA, imageB, imageC])
            print(imageA.shape, imageB.shape, imageC.shape)

            cv.imshow('frame', hstacked)
            cv.waitKey(0)
