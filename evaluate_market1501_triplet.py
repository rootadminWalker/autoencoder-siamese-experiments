import cv2 as cv
import numpy as np
import torch.utils.data
from torchinfo import summary

import utils
from datasets import TripletMarket1501Dataset
from models import Market1501TripletMiniVGG

torch.multiprocessing.set_start_method('spawn', force=True)


def main():
    cap = cv.VideoCapture()
    model = Market1501TripletMiniVGG(
        input_shape=(1, 3, 128, 64),
        embedding_dim=32,
        conv_blocks=2,
    )
    model.load_state_dict(torch.load(
        '/home/rootadminwalker/D/outputs/Market1501_triplet_outputs/model_name(Mini-VGG)_embedding_dim(32)_ep30_loss(triplet)_margin(2)(Early stopped)/model_checkpoints/ep2_ilNone_train-loss0.0657_val-loss0.0477.pth'
    ))
    model.to('cuda:0')
    summary(model, input_shape=[[1, 3, 128, 64]] * 3)
    model.eval()

    with torch.no_grad():
        while cap.isOpened():
            _, frame = cap.read()
            if frame is None:
                continue


            cv.imshow('frame', frame)
            key = cv.waitKey(0) & 0xFF
            if key in [27, ord('q')]:
                break

        cap.release()
        cv.destroyAllWindows()


if __name__ == '__main__':
    main()
