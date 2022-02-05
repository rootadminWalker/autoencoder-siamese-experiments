import os

import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchinfo import summary

from models import OldConvSiamese


class ConvSiamese(OldConvSiamese):
    def __init__(self, embedding_dim):
        super(ConvSiamese, self).__init__(embedding_dim)

    def forward(self, x1, _):
        embedding1 = self.model(x1)
        return embedding1


if __name__ == '__main__':
    device = 'cuda:0'

    checkpoints_path = '/media/rootadminwalker/DATA/outputs/mnist_siamese_outputs/embedding_dim_2_ep100_loss_contrastive_margin_3/model_checkpoints'
    cluster_plots_path = '/media/rootadminwalker/DATA/outputs/mnist_siamese_outputs/embedding_dim_2_ep100_loss_contrastive_margin_3/cluster_plots'
    checkpoints = os.listdir(checkpoints_path)
    sorted_checkpoints = sorted(checkpoints, key=lambda f: int(f.split('_')[0][2:]))
    sorted_checkpoints.append(os.path.join(checkpoints_path, '../siamese.pth'))

    for t, checkpoint in enumerate(sorted_checkpoints):
        model = ConvSiamese(embedding_dim=2)
        model.load_state_dict(torch.load(os.path.join(checkpoints_path, checkpoint)))
        model.eval()
        model.to(device)
        summary(model, input_size=(1, 1, 28, 28))

        test_dataset = torchvision.datasets.MNIST(
            root='/tmp',
            train=False,
            transform=torchvision.transforms.ToTensor(),
            download=True
        )

        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

        color_map = {
            0: 'tab:red',
            1: 'tab:blue',
            2: 'tab:gray',
            3: 'tab:purple',
            4: 'tab:orange',
            5: 'tab:cyan',
            6: 'tab:green',
            7: 'tab:brown',
            8: 'tab:pink',
            9: 'tab:olive'
        }

        plt.style.use("ggplot")
        plt.figure()
        plt.xlim(-6.2, 6.5)
        plt.ylim(-5, 6.9)

        for idx, (image, label) in enumerate(test_dataloader):
            color = color_map[label.cpu().numpy()[0]]
            image = image.cuda()
            embedding = model(image).detach().cpu()
            plt.scatter(*embedding[0], color=color)
            # print(f"Plotted: {idx + 1}")

        epoch = t + 1
        plt.title(f"Siamese network (Epoch {epoch}) cluster plot (embedding_dim=2)")
        plt.xlabel("Embedding dim 1")
        plt.ylabel(f"Embedding dim 2")
        plt.savefig(os.path.join(cluster_plots_path, f'ep{epoch :>03d}'))
        plt.show()
        print(f'Epoch {epoch} completed')
