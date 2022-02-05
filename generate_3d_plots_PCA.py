import os
import pickle

import torch
import torchvision
from sklearn.decomposition import PCA
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
    data = {'points': []}

    checkpoints_path = '/media/rootadminwalker/DATA/outputs/mnist_siamese_outputs/embedding_dim_4_ep70_loss_contrastive_margin_3/model_checkpoints'
    checkpoints = os.listdir(checkpoints_path)
    sorted_checkpoints = sorted(checkpoints, key=lambda f: int(f.split('_')[0][2:]))
    sorted_checkpoints.append(os.path.join(checkpoints_path, '../siamese.pth'))
    pca = PCA(n_components=3)

    for t, checkpoint in enumerate(sorted_checkpoints):
        epoch = t + 1
        data[epoch] = {
            'label_idxes': {
                0: [],
                1: [],
                2: [],
                3: [],
                4: [],
                5: [],
                6: [],
                7: [],
                8: [],
                9: [],
            },
        }
        model = ConvSiamese(embedding_dim=4)
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

        for image, label in test_dataloader:
            label = label.cpu().numpy()[0]
            image = image.cuda()
            embedding = model(image).detach().cpu().numpy()[0]
            data['points'].append(embedding)
            idx = len(data['points']) - 1 if len(data['points']) > 0 else 0
            data[epoch]['label_idxes'][label].append(idx)

        print(f"Epoch {epoch} completed")

    data['points'] = pca.fit_transform(data['points'])
    pickle_path = os.path.join(checkpoints_path, '../PCA_embeddings.pkl')
    with open(pickle_path, 'wb+') as p:
        pickle.dump(data, p)
