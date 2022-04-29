import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import utils

from datasets import SiameseMarket1501Dataset
from models import Market1501TripletMiniVGGEval

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    device = 'cuda:0'
    input_shape = (1, 3, 128, 64)
    model = Market1501TripletMiniVGGEval(
        input_shape=input_shape,
        embedding_dim=32,
        conv_blocks=2,
        conv_kernel_size=(3, 3),
        max_pool_kernel_size=(2, 2),
        dropout_rate=0.03,
        filters=64
    )

    model.load_state_dict(torch.load(
        '/media/rootadminwalker/DATA/outputs/Market1501_triplet_outputs/model_name(Mini-VGG)_embedding_dim(32)_ep30_loss(triplet)_margin(2)/model_checkpoints/ep1_ilNone_train-loss0.1475_val-loss0.2292.pth'))
    model.to(device)
    model.eval()

    train_dataset = SiameseMarket1501Dataset('/media/rootadminwalker/DATA/datasets/Market-1501-v15.09.15', device='cuda:0',
                                       batch_size=32,
                                       pairs_per_image=12)

    test_dataloader = DataLoader(train_dataset, batch_size=32, num_workers=0)

    for batch_labels, batch_images in test_dataloader:
        for idx in range(batch_images.shape[0]):
            imageA = batch_images[idx, 0] / 255
            imageB = batch_images[idx, 1] / 255

            embeddingA = model(imageA)
            embeddingB = model(imageB)

            dist = F.pairwise_distance(embeddingA, embeddingB)
            print(dist)

            imageA = utils.torch_to_cv2(imageA * 255)
            imageB = utils.torch_to_cv2(imageB * 255)

            hstacked = np.hstack([imageA, imageB])
            print(imageA.shape, imageB.shape)

            cv.imshow('frame', hstacked)
            cv.waitKey(0)
