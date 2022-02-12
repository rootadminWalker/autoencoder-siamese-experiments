import argparse
import json
import math
import os
import pickle

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator, StrMethodFormatter
from rich.console import Console
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn, TimeElapsedColumn
from rich.rule import Rule
from rich.table import Table
from rich.traceback import install
from torchinfo import summary
from torchviz import make_dot

from datasets import TripletMNISTLoader
from losses import TripletLoss
from models import ConvTriplet

install()
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
console = Console()
print = console.print


def show_epoch_result_table(epoch, train_loss, val_loss):
    table = Table(title=f"Epoch {epoch} results", show_lines=True, show_edge=True)
    table.add_column('Loss Type')
    table.add_column('Value', justify='right')
    table.add_row('Training loss', train_loss)
    table.add_row('Validation loss', val_loss)
    print(table)


def make_progress_table():
    train_validate_progress = Progress(
        SpinnerColumn('dots'),
        TextColumn("[progress.description]{task.description}"),
        TextColumn("[{task.completed}/{task.total}]"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn()
    )

    epoch_progress = Progress(
        SpinnerColumn('betaWave'),
        TextColumn("[progress.description]{task.description}"),
        TextColumn("[{task.completed}/{task.total}]", style='bold magenta1'),
        BarColumn(bar_width=150),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn()
    )

    progress_table = Table.grid()
    progress_table.add_row(train_validate_progress)
    progress_table.add_row(epoch_progress)

    return progress_table, train_validate_progress, epoch_progress


def validate(progress, validation_task, device, dataloader, model, loss_fn, batch_size):
    with torch.no_grad():
        loss_history = []
        model.eval()
        for batch, pair_images in enumerate(dataloader.validate()):
            anchors, positives, negatives = pair_images

            # Compute prediction error
            ap_dist, an_dist = model(anchors, positives, negatives)
            loss = loss_fn(ap_dist, an_dist)
            loss_history.append(loss)

            loss, current = loss.item(), batch * batch_size
            progress.console.log(f"val_loss: {loss:>7f}")

            progress.update(
                validation_task,
                description=f'[red]Validating...',
                advance=batch_size
            )

        val_loss = torch.tensor(loss_history)
        return val_loss


def train(epoch, progress, device, dataloader, model, loss_fn, optimizer, batch_size):
    train_loss_history = []
    model.train()

    train_task = progress.add_task(f"[yellow]Waiting to train", total=dataloader.train_len)
    validation_task = progress.add_task("[yellow]Validation...waiting", total=dataloader.valid_len, start=False)

    print(Rule('Training stage'))
    for batch, pair_images in enumerate(dataloader.train()):
        anchors, positives, negatives = pair_images

        # Zero gradient everything
        optimizer.zero_grad()

        # Compute prediction error
        ap_dist, an_dist = model(anchors, positives, negatives)
        loss = loss_fn(ap_dist, an_dist)
        train_loss_history.append(loss)

        # Backpropagation
        loss.backward()
        optimizer.step()

        loss, current = loss.item(), batch * batch_size
        progress.console.log(f"loss: {loss:>7f}")

        progress.update(
            train_task,
            description=f"[red]Training...",
            advance=pair_images.shape[1]
        )

    train_loss_history = torch.tensor(train_loss_history)

    progress.update(train_task, description=f"[green]Training stage...completed", )
    print(Rule("Validation round"))
    progress.start_task(validation_task)
    val_loss = validate(
        progress=progress,
        validation_task=validation_task,
        device=device,
        dataloader=dataloader,
        model=model,
        loss_fn=loss_fn,
        batch_size=batch_size,
    ).detach().mean().cpu().numpy()
    progress.update(validation_task, description=f"[green]Validation stage...completed")
    print(Rule(f'Mean val_loss: {val_loss}'))

    progress.remove_task(train_task)
    progress.remove_task(validation_task)

    show_epoch_result_table(
        epoch=epoch,
        train_loss=str(train_loss_history.mean().detach().cpu().numpy()),
        val_loss=str(val_loss)
    )

    return train_loss_history, val_loss


def plot_graph_per_steps(fn, train_history, val_history, train_steps, loss_fn_name):
    TN = np.arange(0, len(train_history))
    VN = np.arange(train_steps, len(train_history) + 1, train_steps)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(TN, train_history, label="train_loss")
    plt.plot(VN, val_history, label="val_loss")
    plt.scatter(VN, val_history, c='#1f77b4', zorder=3)
    plt.title("Training and Validation Loss (Steps)")
    plt.xlabel("Steps #")
    plt.ylabel(f"Loss ({loss_fn_name})")
    plt.legend(loc="upper right")
    plt.savefig(fn)
    # plt.show()


def plot_graph_per_epoch(fn, epochs, train_history, val_history, train_steps, loss_fn_name):
    train_history_per_epoch = []
    for idx in np.arange(0, epochs):
        epoch_loss = np.mean(train_history[idx * train_steps:(idx + 1) * train_steps])
        train_history_per_epoch.append(epoch_loss)
    N = np.arange(0, epochs)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, train_history_per_epoch, label="train_loss")
    plt.plot(N, val_history, label="val_loss")
    plt.scatter(N, val_history, c='#1f77b4', zorder=3)

    plt.gca().xaxis.set_major_locator(MultipleLocator(2))
    plt.gca().xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))

    plt.title("Training and Validation Loss (Epochs)")
    plt.xlabel("Epochs #")
    plt.ylabel(f"Loss ({loss_fn_name})")
    plt.legend(loc="upper right")
    plt.savefig(fn)
    # plt.show()


def plot_graphs(epochs, train_history, val_history, train_steps, loss_fn_name, fn1, fn2):
    plot_graph_per_steps(fn1, train_history, val_history, train_steps, loss_fn_name)
    plt.clf()
    plot_graph_per_epoch(fn2, epochs, train_history, val_history, train_steps, loss_fn_name)


def main(args):
    epochs = args['epochs']
    batch_size = args['batch_size']
    opt = args['optimizer']
    loss_margin = args['margin']
    loss_fn_name = args['loss']
    image_limit = args['image_limit']
    device = args['device']
    embedding_dim = args['embedding_dim']
    conv_blocks = args['conv_blocks']
    filters = args['filters']
    output_dir = args['output_dir']

    optimizers_map = {
        'adam': torch.optim.Adam,
        'SGD': torch.optim.SGD
    }
    loss_map = {
        'triplet': TripletLoss
    }

    model = ConvTriplet(embedding_dim=embedding_dim, conv_blocks=conv_blocks, filters=filters)
    summary(model, input_size=((1, 1, 28, 28), (1, 1, 28, 28), (1, 1, 28, 28)))
    model.to(device)

    transforms = T.Compose([
        T.ToTensor(),
        T.Lambda(lambda i: i.to(device))
    ])

    dataset = torchvision.datasets.MNIST(
        root='/tmp/',
        train=True,
        transform=transforms,
        download=True,
    )

    train_history = []
    val_history = []

    dataloader = TripletMNISTLoader(
        dataset,
        device=device,
        batch_size=batch_size,
        train_valid_split=0.9,
        image_limit=image_limit,
    )

    train_steps = math.ceil(dataloader.train_len / batch_size) + 1

    output_dir_name = f"embedding_dim_{embedding_dim}_ep{epochs}_loss_{args['loss']}_margin_{loss_margin}"
    output_base_path = os.path.join(output_dir, output_dir_name)
    model_checkpoints_path = os.path.join(output_base_path, 'model_checkpoints')
    history_checkpoints_path = os.path.join(output_base_path, 'history_checkpoints')
    model_structures_path = os.path.join(output_base_path, 'model_structures')
    info_file_path = os.path.join(output_base_path, 'info.json')
    loss_steps_path = os.path.join(output_base_path, 'loss_steps.png')
    loss_epochs_path = os.path.join(output_base_path, 'loss_epochs.png')

    if not os.path.exists(output_base_path):
        os.mkdir(output_base_path)
        os.mkdir(model_checkpoints_path)
        os.mkdir(model_structures_path)
        os.mkdir(history_checkpoints_path)

    make_dot(torch.tensor(0.5), params=dict(list(model.named_parameters()))).render(
        os.path.join(model_structures_path, 'siamese.png'), format='png')

    siamese_output_path = os.path.join(output_base_path, 'siamese.pth')

    opt = optimizers_map[opt](model.parameters(), lr=args['lr'])
    loss_func = loss_map[loss_fn_name](margin=loss_margin)

    progress_table, train_validation_progress, epoch_progress = make_progress_table()
    epoch_task = epoch_progress.add_task('[bold dark_slate_gray1]Epoch', completed=1, total=epochs)
    with Live(progress_table, refresh_per_second=10):
        for t in range(epochs):
            current_epoch = t + 1
            print(Rule(f"Epoch {current_epoch}"))
            train_loss_history, val_loss = train(
                epoch=current_epoch,
                progress=train_validation_progress,
                device=device,
                dataloader=dataloader,
                model=model,
                loss_fn=loss_func,
                optimizer=opt,
                batch_size=batch_size,
            )
            train_history.extend(train_loss_history)
            val_history.append(val_loss)
            if current_epoch < epochs:
                checkpoint_model_path = os.path.join(
                    model_checkpoints_path,
                    f'ep{current_epoch}_il{image_limit}_train-loss{train_loss_history.mean():.4f}_val-loss{val_loss:.4f}.pth')
                torch.save(model.state_dict(), checkpoint_model_path)

                history_checkpoint_file = os.path.join(history_checkpoints_path, f'ep{current_epoch}.pickle')
                with open(history_checkpoint_file, 'wb+') as f:
                    pickle.dump({'train_history': train_history, 'val_history': val_history}, f)

            epoch_progress.update(epoch_task, advance=1)

    torch.save(model.state_dict(), siamese_output_path)

    plot_graphs(
        epochs=epochs,
        train_history=train_history,
        val_history=val_history,
        train_steps=train_steps,
        loss_fn_name=args['loss'],
        fn1=loss_steps_path,
        fn2=loss_epochs_path
    )

    with open(info_file_path, 'w+') as f:
        json.dump({
            'trained_epochs': epochs,
            'batch_size': batch_size,
            'loss_func': args['loss'],
            'margin': loss_margin,
            'optimizer': opt,
            'image_amount': image_limit,
        }, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output-dir', type=str, required=True,
                        help="Output dir for the training result, includes tensorboard logs and model checkpoints")
    parser.add_argument('-e', '--epochs', type=int, required=True,
                        help="Epochs of training")
    parser.add_argument('-b', '--batch-size', type=int, default=32,
                        help="Batch size for training")
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='Optimizer for the training')
    parser.add_argument('-lo', '--loss', type=str, default='mse',
                        help='Loss function for training')
    parser.add_argument('-m', '--margin', type=int, default=2,
                        help='Margin of the two loss functions')
    parser.add_argument('-lr', type=float, default=1e-3,
                        help='Learning rate for the training')
    parser.add_argument('-cb', '--conv-blocks', type=int, default=2,
                        help="Conv blocks of the network")
    parser.add_argument('-f', '--filters', type=int, default=64,
                        help='Number of filters of the network')
    parser.add_argument('-il', '--image-limit', type=int, default=None,
                        help='Image limit per epoch')
    parser.add_argument('--embedding-dim', type=int, default=48,
                        help='Num of latent space representation dims')
    parser.add_argument('-d', '--device', type=str, default='cuda:0',
                        help="Device to train the model")

    args = vars(parser.parse_args())
    main(args)
