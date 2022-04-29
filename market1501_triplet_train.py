import argparse
import json
import os
import pickle

import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as T
from rich.console import Console
from rich.live import Live
from rich.rule import Rule
from rich.traceback import install
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

import datasets
import utils
from losses import TripletLoss
from models import Market1501TripletMiniVGG
from utils import make_progress_table, show_epoch_result_table, plot_graphs

torch.multiprocessing.set_start_method('spawn', force=True)

install()
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
console = Console()
print = console.print


def load_model(
        device,
        input_shape,
        embedding_dim,
        conv_blocks,
        conv_kernel_size,
        max_pool_kernel_size,
        dropout_rate,
        filters
):
    model_instance = Market1501TripletMiniVGG
    model_name = model_instance.MODEL_NAME
    model = model_instance(
        input_shape,
        embedding_dim=embedding_dim,
        conv_blocks=conv_blocks,
        conv_kernel_size=conv_kernel_size,
        max_pool_kernel_size=max_pool_kernel_size,
        dropout_rate=dropout_rate,
        filters=filters
    )
    summary(model, input_size=[input_shape] * 3)
    model.to(device)

    return model_name, model


def validate(progress, validation_task, valid_dataloader, model, loss_fn) -> np.array:
    with torch.no_grad():
        val_loss_history = []
        model.eval()
        for batch, triplets in enumerate(valid_dataloader):
            anchors = triplets[:, 0]
            positives = triplets[:, 1]
            negatives = triplets[:, 2]

            # Compute prediction error
            ap_dist, an_dist = model(anchors, positives, negatives)
            val_loss = loss_fn(ap_dist, an_dist)

            valid_batch_size = valid_dataloader.batch_size
            val_loss, current = val_loss.item(), batch * valid_batch_size
            progress.console.log(f"val_loss: {val_loss:>7f}")
            val_loss_history.append(val_loss)

            progress.update(
                validation_task,
                description=f'[red]Validating...',
                advance=valid_batch_size
            )

        val_loss = np.array(val_loss_history).mean()
        return val_loss


def train(epoch, progress, train_dataloader, valid_dataloader, model, loss_fn, optimizer) -> (np.array, np.array):
    train_loss_history = []

    train_batch_size = train_dataloader.batch_size
    train_task = progress.add_task(
        f"[yellow]Waiting to train",
        total=len(train_dataloader) * train_batch_size
    )
    validation_task = progress.add_task(
        "[yellow]Validation...waiting",
        total=len(valid_dataloader) * valid_dataloader.batch_size,
        start=False
    )

    print(Rule('Training stage'))
    for batch, triplets in enumerate(train_dataloader):
        anchors = triplets[:, 0]
        positives = triplets[:, 1]
        negatives = triplets[:, 2]

        # Zero gradient everything
        optimizer.zero_grad()

        # Compute prediction error
        ap_dist, an_dist = model(anchors, positives, negatives)
        loss = loss_fn(ap_dist, an_dist)

        # Backpropagation
        loss.backward()
        optimizer.step()

        loss, current = loss.item(), batch * train_batch_size
        progress.console.log(f"loss: {loss:>7f}")

        train_loss_history.append(loss)

        progress.update(
            train_task,
            description=f"[red]Training...",
            advance=train_batch_size
        )

    progress.update(train_task, description=f"[green]Training stage...completed", )
    print(Rule("Validation round"))

    progress.start_task(validation_task)
    val_loss = validate(
        progress=progress,
        validation_task=validation_task,
        valid_dataloader=valid_dataloader,
        model=model,
        loss_fn=loss_fn,
    )
    progress.update(validation_task, description=f"[green]Validation stage...completed")
    print(Rule(f'Mean val_loss: {val_loss}'))

    progress.remove_task(train_task)
    progress.remove_task(validation_task)

    show_epoch_result_table(
        epoch=epoch,
        train_loss=str(np.mean(train_loss_history)),
        val_loss=str(val_loss)
    )

    return np.array(train_loss_history), val_loss


def main(args):
    # Get all hyperparameters
    dataset_path = args['dataset_path']
    output_dir = args['output_dir']
    embedding_dim = args['embedding_dim']
    device = args['device']
    input_shape = (1, 3, *list(map(int, args['input_size'].split('x'))))
    epochs = args['epochs']
    inital_epoch = args['initial_epoch']
    batch_size = args['batch_size']
    loss_name = args['loss']
    loss_margin = args['margin']
    opt_name = args['optimizer']
    lr = args['lr']
    conv_blocks = args['conv_blocks']
    filters = args['filters']
    conv_kernel_size = (args['conv_kernel_size'],) * 2
    max_pool_kernel_size = (args['max_pool_kernel_size'],) * 2
    dropout_rate = args['dropout_rate']
    image_limit = args['image_limit']
    triplets_per_anchor = args['triplets_per_anchor']
    train_valid_split = args['train_valid_split']
    num_workers = args['num_workers']
    pin_memory = args['pin_memory']
    use_cudnn_autotuner = args['use_cudnn_autotuner']

    # Define the loss and optimizer map by parsing according to the hyperparameters
    optimizers_map = {
        'adam': torch.optim.Adam,
        'SGD': torch.optim.SGD
    }
    loss_map = {
        'triplet': TripletLoss
    }

    # Use cudnn's autotuner if user wants to
    if use_cudnn_autotuner:
        torch.backends.cudnn.benchmark = True

    # Load the model
    model_name, model = load_model(
        device=device,
        input_shape=input_shape,
        embedding_dim=embedding_dim,
        conv_blocks=conv_blocks,
        conv_kernel_size=conv_kernel_size,
        max_pool_kernel_size=max_pool_kernel_size,
        dropout_rate=dropout_rate,
        filters=filters
    )

    # Load the dataset
    transforms = T.Compose([
        # T.Lambda(lambda x: x / 255),
        T.Resize(size=(input_shape[-1], input_shape[-2]))
    ])
    dataset = datasets.TripletMarket1501Dataset(
        dataset_path,
        device=device,
        batch_size=batch_size,
        transforms=transforms,
        image_limit=image_limit,
        triplets_per_anchor=triplets_per_anchor,
    )
    dataset_len = len(dataset)

    # Split dataset into train and valid
    train_dataset, valid_dataset = utils.train_valid_split(dataset, train_valid_split=train_valid_split)

    # Initialize the dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True
    )

    train_history = []
    val_history = []
    train_steps_per_epoch = len(train_dataloader)

    # Get all absolute paths of required directories
    output_dir_name = f"model_name({model_name})_embedding_dim({embedding_dim})_ep{epochs}_loss({args['loss']})_margin({loss_margin})"
    output_base_path = os.path.join(output_dir, output_dir_name)
    model_checkpoints_path = os.path.join(output_base_path, 'model_checkpoints')
    history_checkpoints_path = os.path.join(output_base_path, 'history_checkpoints')
    model_structures_path = os.path.join(output_base_path, 'model_structures')
    tensorboard_logs_path = os.path.join(output_base_path, 'tensorboard_logs')
    info_file_path = os.path.join(output_base_path, 'info.json')
    loss_steps_path = os.path.join(output_base_path, 'loss_steps.png')
    loss_epochs_path = os.path.join(output_base_path, 'loss_epochs.png')
    triplet_output_path = os.path.join(output_base_path, 'triplet.pth')
    os.path.join(model_structures_path, 'triplet.png')

    # If the base path doesn't exist, create everything
    if not os.path.exists(output_base_path):
        os.mkdir(output_base_path)
        os.mkdir(model_checkpoints_path)
        os.mkdir(model_structures_path)
        os.mkdir(history_checkpoints_path)
        os.mkdir(tensorboard_logs_path)

    # Initialize the log writer
    log_writer = SummaryWriter(log_dir=tensorboard_logs_path, flush_secs=3)

    # Save the structure of the model
    utils.make_model_structure(
        model=model,
        input_shape=input_shape,
        device=device,
        output_path=os.path.join(model_structures_path, 'triplet.png'),
        image_format='png'
    )

    # Get the optimizer and loss module from the map
    opt = optimizers_map[opt_name](model.parameters(), lr=lr)
    loss_func = loss_map[loss_name](margin=loss_margin)

    # Define the progress tables for visualization
    progress_table, train_validation_progress, epoch_progress = make_progress_table()
    epoch_task = epoch_progress.add_task('[bold dark_slate_gray1]Epoch', completed=1, total=epochs)
    model.train()
    with Live(progress_table, refresh_per_second=10):
        # Start training
        for t in range(inital_epoch, epochs):
            current_epoch = t + 1
            # Dump info every epoch
            with open(info_file_path, 'w+') as f:
                json.dump({
                    'model_name': model_name,
                    'input_size': args['input_size'],
                    'trained_epochs': current_epoch,
                    'early_stopped': True,
                    'batch_size': batch_size,
                    'loss_func': loss_name,
                    'margin': loss_margin,
                    'optimizer': opt_name,
                    'image_amount': dataset_len,
                    'conv_blocks': conv_blocks,
                    'conv_kernel_size': conv_kernel_size,
                    'max_pool_kernel_size': max_pool_kernel_size,
                    'dropout_rate': dropout_rate,
                    'filters': filters,
                    'num_workers': num_workers,
                    'pin_memory': pin_memory,
                    'use_cudnn_autotuner': use_cudnn_autotuner
                }, f, indent=4)
            print(Rule(f"Epoch {current_epoch}"))
            # Train func
            train_loss_per_step, val_loss = train(
                epoch=current_epoch,
                progress=train_validation_progress,
                train_dataloader=train_dataloader,
                valid_dataloader=valid_dataloader,
                model=model,
                loss_fn=loss_func,
                optimizer=opt,
            )
            train_history.extend(train_loss_per_step)
            val_history.append(val_loss)

            # Log the instant data on tensorboard
            utils.log_train_valid_losses(
                log_writer=log_writer,
                current_epoch=current_epoch,
                train_steps_per_epoch=train_steps_per_epoch,
                train_loss_per_step=train_loss_per_step,
                val_loss=val_loss
            )

            # Save the checkpoint models
            if current_epoch < epochs:
                checkpoint_model_path = os.path.join(
                    model_checkpoints_path,
                    f'ep{current_epoch}_il{image_limit}_train-loss{train_loss_per_step.mean():.4f}_val-loss{val_loss:.4f}.pth')
                torch.save(model.state_dict(), checkpoint_model_path)

                history_checkpoint_file = os.path.join(history_checkpoints_path, f'ep{current_epoch}.pickle')
                with open(history_checkpoint_file, 'wb+') as f:
                    pickle.dump({'train_history': train_history, 'val_history': val_history}, f)

            epoch_progress.update(epoch_task, advance=1)

    torch.save(model.state_dict(), triplet_output_path)
    with open(info_file_path, 'w+') as f:
        json.dump({
            'model_name': model_name,
            'input_size': args['input_size'],
            'trained_epochs': epochs,
            'initial_epoch': inital_epoch,
            'batch_size': batch_size,
            'loss_func': loss_name,
            'margin': loss_margin,
            'optimizer': opt_name,
            'image_amount': dataset_len,
            'conv_blocks': conv_blocks,
            'conv_kernel_size': conv_kernel_size,
            'max_pool_kernel_size': max_pool_kernel_size,
            'dropout_rate': dropout_rate,
            'filters': filters,
            'num_workers': num_workers,
            'pin_memory': pin_memory,
            'use_cudnn_autotuner': use_cudnn_autotuner
        }, f, indent=4)

    plot_graphs(
        epochs=epochs,
        train_history=train_history,
        val_history=val_history,
        train_steps=train_steps_per_epoch,
        loss_name=loss_name,
        fn1=loss_steps_path,
        fn2=loss_epochs_path
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--dataset-path', type=str, required=True,
                        help='Path to the Market1501 dataset')
    parser.add_argument('-o', '--output-dir', type=str, required=True,
                        help="Output dir for the training result, includes tensorboard logs and model checkpoints")
    parser.add_argument('--embedding-dim', type=int, required=True,
                        help='Num of latent space representation dims')
    parser.add_argument('-d', '--device', type=str, default='cuda:0',
                        help="Device to train the model")
    parser.add_argument('--input-size', type=str, default='128x64', required=True,
                        help='Input size for the model, format is "hxw"')
    parser.add_argument('-e', '--epochs', type=int, required=True,
                        help="Epochs of training")
    parser.add_argument('--initial-epoch', type=int, default=0,
                        help="Start epoch of the training")
    parser.add_argument('-b', '--batch-size', type=int, default=32,
                        help="Batch size for training")
    parser.add_argument('-lo', '--loss', type=str, default='triplet',
                        help='Loss function for training')
    parser.add_argument('-m', '--margin', type=int, default=2,
                        help='Margin of the loss function')
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='Optimizer for the training')
    parser.add_argument('-lr', type=float, default=1e-3,
                        help='Learning rate for the training')
    parser.add_argument('-cb', '--conv-blocks', type=int, default=2,
                        help="Conv blocks of the network")
    parser.add_argument('-f', '--filters', type=int, default=64,
                        help='Number of filters of the network')
    parser.add_argument('--conv-kernel-size', type=int, required=True,
                        help="Convolution kernel size of the model, will be (this_parameter x this_parameter)")
    parser.add_argument('--max-pool-kernel-size', type=int, required=True,
                        help="Max pool kernel size of the model (this_parameter x this_parameter)")
    parser.add_argument('--dropout-rate', type=float, required=True,
                        help="Dropout rate of the model")
    parser.add_argument('-il', '--image-limit', type=int, default=None,
                        help='Image limit per epoch')
    parser.add_argument('--triplets-per-anchor', type=int, default=None,
                        help="How many triplets per anchor")
    parser.add_argument('--train-valid-split', type=float, default=0.8,
                        help="Train-valid of the dataset")
    parser.add_argument('--num-workers', type=int, default=0,
                        help="num_workers parameter of torch.utils.data.DataLoader")
    parser.add_argument('--pin-memory', type=bool, default=False,
                        help="Set the pin_memory parameter of torch.utils.data.DataLoader")
    parser.add_argument('--use-cudnn-autotuner', type=bool, default=False,
                        help="If to use cudnn's autotuner to boost speed")

    args = vars(parser.parse_args())
    main(args)
