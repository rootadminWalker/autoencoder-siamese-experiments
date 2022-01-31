import argparse
import json
import os
from random import choice

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchinfo import summary
import torch.nn.functional as F
from torchviz import make_dot

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class ContrastiveLoss(nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()

    @staticmethod
    def contrastive_loss(y, Dw, margin=2):
        return torch.mean((1 - y) * Dw.pow(2) + (y) * torch.pow(torch.clamp(margin - Dw, min=0.0), 2))

    def forward(self, embedding1, embedding2, label):
        dist = F.pairwise_distance(embedding1, embedding2, keepdim=True)
        loss = self.contrastive_loss(label, dist)
        return loss


class ConvSiamese(nn.Module):
    def __init__(self, embedding_dim):
        super(ConvSiamese, self).__init__()
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


class SiameseMNISTLoader(DataLoader):
    def __init__(self, mnist_dataset, batch_size):
        super(SiameseMNISTLoader, self).__init__(None, batch_size=batch_size, num_workers=4, pin_memory=True)
        self.mnist_dataset = mnist_dataset
        self.full_length = len(self.mnist_dataset) * self.batch_size

    def __len__(self):
        return len(self.mnist_dataset)

    def __iter__(self):
        for idx in range(len(self)):
            selected_im = self.mnist_dataset[idx][0]
            selected_label = self.mnist_dataset[idx][1]

            true_label_idxes = [i for i in range(len(self.mnist_dataset)) if
                                self.mnist_dataset.targets[i] == selected_label]
            false_label_idxes = [i for i in range(len(self.mnist_dataset)) if
                                 self.mnist_dataset.targets[i] != selected_label]

            pair_images, pair_labels = [], []

            for b in range(self.batch_size // 2):
                # Positive pair
                true_pair_idx = choice(true_label_idxes)
                true_image_b = self.mnist_dataset[true_pair_idx][0]
                true_image_pair = [selected_im, true_image_b]
                pair_images.append(torch.stack(true_image_pair))
                pair_labels.append(1)

                # Negative pair
                false_pair_idx = choice(false_label_idxes)
                false_image_b = self.mnist_dataset[false_pair_idx][0]
                false_image_pair = [selected_im, false_image_b]
                pair_images.append(torch.stack(false_image_pair))
                pair_labels.append(0)

            pair_images = torch.stack(pair_images).reshape((2, self.batch_size, 1, 28, 28))
            yield pair_images, torch.tensor(pair_labels)


def train(device, dataloader, model, loss_fn, optimizer, batch_size, image_limit=None):
    size = dataloader.full_length
    if image_limit is None:
        image_limit = size

    model.train()
    for batch, (pair_images, labels) in enumerate(dataloader):
        imagesA, imagesB = pair_images
        imagesA = imagesA.to(device)
        imagesB = imagesB.to(device)
        labels = labels.to(device)
        # print(batch)

        optimizer.zero_grad()

        # Compute prediction error
        dist = model(imagesA, imagesB)
        loss = loss_fn(dist, (1 - labels).abs().type(torch.float)).sum()

        # Backpropagation
        loss.backward()
        optimizer.step()

        loss, current = loss.item(), batch * batch_size
        print(f"loss: {loss:>7f} [{current:>5d}/{image_limit:>5d}]")

        if current >= image_limit:
            break


def main(args):
    optimizers_map = {
        'adam': torch.optim.Adam,
        'SGD': torch.optim.SGD
    }
    device = args['device']

    model = ConvSiamese(embedding_dim=args['embedding_dim'])
    summary(model, input_size=((1, 1, 28, 28), (1, 1, 28, 28)))
    model.to(device)

    train_dataset = torchvision.datasets.MNIST(
        root='/tmp/',
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )

    # test_dataset = torchvision.datasets.MNIST(
    #     root='/tmp',
    #     train=False,
    #     transform=torchvision.transforms.ToTensor(),
    #     download=True
    # )

    EPOCHS = args['epochs']
    BATCH_SIZE = args['batch_size']
    OPT = args['optimizer']

    train_dataloader = SiameseMNISTLoader(train_dataset, batch_size=BATCH_SIZE)

    output_base_path = args['output_path']
    model_checkpoints_path = os.path.join(output_base_path, 'model_checkpoints')
    model_structures_path = os.path.join(output_base_path, 'model_structures')
    info_file_path = os.path.join(output_base_path, 'info.json')

    if not os.path.exists(output_base_path):
        os.mkdir(output_base_path)
        os.mkdir(model_checkpoints_path)
        os.mkdir(model_structures_path)

    make_dot(torch.tensor(0.5), params=dict(list(model.named_parameters()))).render(
        os.path.join(model_structures_path, 'siamese.png'), format='png')

    siamese_output_path = os.path.join(output_base_path, 'siamese.pth')

    opt = optimizers_map[OPT](model.parameters(), lr=args['lr'])
    for t in range(EPOCHS):
        print(f"Epoch {t + 1}\n-------------------------------------")
        train(device, train_dataloader, model, nn.MSELoss(), opt, BATCH_SIZE)

    torch.save(model.state_dict(), siamese_output_path)

    # N = np.arange(0, EPOCHS)
    # plt.style.use("ggplot")
    # plt.figure()
    # plt.plot(N, history.history["loss"], label="train_loss")
    # plt.plot(N, history.history["val_loss"], label="val_loss")
    # plt.plot(N, history.history["accuracy"], label="train_acc")
    # plt.plot(N, history.history["val_accuracy"], label="val_acc")
    # plt.title("Training Loss and Accuracy")
    # plt.xlabel("Epoch #")
    # plt.ylabel("Loss/Accuracy")
    # plt.legend(loc="lower left")

    with open(info_file_path, 'w+') as f:
        json.dump({
            'trained_epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'loss_func': 'contrastive_loss',
            'optimizer': OPT
        }, f, indent=4)

    # model.eval()
    # for batch_images, batch_labels in train_dataloader:
    #     batch_images = batch_images.reshape((32, 2, 1, 28, 28))
    #     for pair_image, label in zip(batch_images, batch_labels):
    #         imageA, imageB = pair_image
    #         imageA = torch.stack([imageA]).to(device)
    #         imageB = torch.stack([imageB]).to(device)
    #         dist = model(imageA, imageB).cpu()
    #         print(f'Estimated Dist: {dist}, Loss: {model.contrastive_loss(label, dist)}')
    #
    #         imageA = np.array(imageA.cpu()).reshape((28, 28, 1)) * 255
    #         imageA = imageA.astype('uint8')
    #         imageB = np.array(imageB.cpu()).reshape((28, 28, 1)) * 255
    #         imageB = imageB.astype('uint8')
    #         hstacked = np.hstack([imageA, imageB])
    #         print(label)
    #
    #         cv.imshow('frame', hstacked)
    #         cv.waitKey(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output-path', type=str, required=True,
                        help="Output path for the training result, includes tensorboard logs and model checkpoints")
    parser.add_argument('-e', '--epochs', type=int, required=True,
                        help="Epochs of training")
    parser.add_argument('-b', '--batch-size', type=int, default=32,
                        help="Batch size for training")
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='Optimizer for the training')
    parser.add_argument('-lr', type=float, default=1e-3,
                        help='Learning rate for the training')
    parser.add_argument('--embedding-dim', type=int, default=48,
                        help='Num of latent space representation dims')
    parser.add_argument('-d', '--device', type=str, default='cuda:0',
                        help="Device to train the model")

    args = vars(parser.parse_args())
    main(args)
