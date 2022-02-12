import cv2 as cv
import numpy as np
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

from datasets import TripletMNISTLoader
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
    device = 'cuda:0'
    # model = ConvSiamese(embedding_dim=48)
    # model.load_state_dict(torch.load(
    #     '/media/rootadminwalker/DATA/outputs/mnist_siamese_outputs/embedding_dim_48_ep15_loss_contrastive_margin_3/siamese.pth'))
    # model.eval()
    # summary(model, input_size=((1, 1, 28, 28), (1, 1, 28, 28)))
    # model.to(device)

    transforms = T.Compose([
        T.ToTensor(),
        T.Lambda(lambda i: i.cuda())
    ])

    test_dataset = torchvision.datasets.MNIST(
        root='/tmp',
        train=True,
        transform=transforms,
        download=True
    )
    test_dataloader = TripletMNISTLoader(test_dataset, device='cuda:0', batch_size=32, train_valid_split=0.9,
                                         image_limit=50, pairs_per_anchor=3)

    # for batch_images, batch_labels in test_dataloader.validate():
    for batch_images in test_dataloader.train():
        print(batch_images.shape)
        for idx in range(batch_images.shape[1]):
            imageA = batch_images[0, idx].cuda().reshape(1, 1, 28, 28)
            imageB = batch_images[1, idx].cuda().reshape(1, 1, 28, 28)
            imageC = batch_images[2, idx].cuda().reshape(1, 1, 28, 28)
            # label = batch_labels[idx].cuda()
            # dist, embedding1, embedding2 = model(imageA, imageB)

            imageA = np.array(imageA.cpu()).reshape((28, 28, 1)) * 255
            imageA = imageA.astype('uint8')
            imageB = np.array(imageB.cpu()).reshape((28, 28, 1)) * 255
            imageB = imageB.astype('uint8')
            imageC = np.array(imageC.cpu()).reshape((28, 28, 1)) * 255
            imageC = imageC.astype('uint8')
            hstacked = np.hstack([imageA, imageB, imageC])

            # # loss = torch.nn.MSELoss()((2 - label).abs().type(torch.float).cuda(), dist)
            # cpu_dist = dist.detach().cpu().numpy()[0]
            # print(f'Estimated Dist: {cpu_dist:.6f}, True label: {label}')
            # print(f'Embedding1: {embedding1}, Embedding2: {embedding2}')
            #
            # hstacked = cv.resize(hstacked, (512, 256))
            # hstacked = cv.putText(hstacked, f'Distance: {cpu_dist:.6f}', (100, 40), cv.FONT_HERSHEY_SIMPLEX,
            #                       1, (255, 255, 255), 2, cv.LINE_AA)

            cv.imshow('frame', hstacked)
            cv.waitKey(0)
