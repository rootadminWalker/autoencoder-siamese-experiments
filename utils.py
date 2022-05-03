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
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
from matplotlib.ticker import MultipleLocator, StrMethodFormatter
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn, TimeElapsedColumn
from rich.table import Table
from torchviz import make_dot

print = Console().print


def require_input(require_func):
    def require_wrapper(check_param_name):
        def decorator(func):
            def ultimate_wrapper(*args, **kwargs):
                arg_pos = func.__code__.co_varnames.index(check_param_name)
                if check_param_name not in kwargs:
                    param = args[arg_pos]
                else:
                    param = kwargs[check_param_name]

                require_func(func.__name__, param, check_param_name)
                return func(*args, **kwargs)
            return ultimate_wrapper
        return decorator
    return require_wrapper


@require_input
def require_cv2_input(func_name, param, check_param_name):
    assert (
        isinstance(param, np.ndarray) or isinstance(param, torch.Tensor)
    ), f"Function '{func_name}' has a parameter {check_param_name} requires a cv2 image, \nbut you have {type(param)}"


@require_input
def require_torch_input(func_name, param, check_param_name):
    assert (
        isinstance(param, np.ndarray) or isinstance(param, torch.Tensor)
    ), f"Function '{func_name}' has a parameter {check_param_name} requires a torch tensor, \nbut you have {type(param)}"


@require_input
def require_cv2_or_torch(func_name, param, check_param_name):
    assert (
        isinstance(param, np.ndarray) or isinstance(param, torch.Tensor)
    ), f"""Function '{func_name}' has a parameter {check_param_name} requires a torch tensor or a cv2 image, 
            but you have {type(param)}"""


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


def show_epoch_result_table(epoch, train_loss, val_loss):
    table = Table(title=f"Epoch {epoch} results", show_lines=True, show_edge=True)
    table.add_column('Loss Type')
    table.add_column('Value', justify='right')
    table.add_row('Training mean loss', train_loss)
    table.add_row('Validation loss', val_loss)
    print(table)


def plot_graph_per_steps(fn, epochs, train_history, val_history, train_steps, loss_fn_name):
    TN = np.arange(0, len(train_history))
    VN = np.arange(train_steps, int(train_steps * (epochs + 1)), train_steps)
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


def plot_graphs(epochs, train_history, val_history, train_steps, loss_name, fn1, fn2):
    plot_graph_per_steps(fn1, epochs, train_history, val_history, train_steps, loss_name)
    plt.clf()
    plot_graph_per_epoch(fn2, epochs, train_history, val_history, train_steps, loss_name)


def torch_to_cv2(tensor):
    assert isinstance(tensor, torch.Tensor), "You decided to 'torch_to_cv2', but not giving me a tensor?\n Hilarious"
    image = tensor.squeeze().permute(1, 2, 0).cpu().numpy().astype('uint8')
    return image


def cv2_to_torch(cv2_image, device='cpu'):
    assert isinstance(cv2_image,
                      np.ndarray), "You decided to 'cv2_to_torch', but not giving me a cv2 image?\n Hilarious"
    return torch.tensor(cv2_image, device=device).permute(2, 0, 1).unsqueeze(0).float()


def train_valid_split(dataset, train_valid_split):
    dataset_len = len(dataset)
    full_indices = range(dataset_len)
    train_indices = random.sample(range(dataset_len), int(dataset_len * train_valid_split))
    train_dataset = torch.utils.data.Subset(dataset, indices=train_indices)

    valid_indices = np.setdiff1d(full_indices, train_indices)
    valid_dataset = torch.utils.data.Subset(dataset, indices=valid_indices)

    return train_dataset, valid_dataset


def log_train_valid_losses(log_writer, current_epoch, train_steps_per_epoch, train_loss_per_step, val_loss):
    for idx in range(train_steps_per_epoch):
        log_writer.add_scalar('Loss/steps', train_loss_per_step[idx],
                              ((current_epoch - 1) * train_steps_per_epoch) + idx)
    log_writer.add_scalars('Loss/epochs', {
        'train_loss': np.mean(train_loss_per_step),
        'val_loss': val_loss
    }, current_epoch)


def make_model_structure(model, input_shape, device, output_path, image_format):
    with torch.no_grad():
        model.eval()
        random_input1 = torch.rand(tuple(input_shape)).to(device)
        random_input2 = torch.rand(tuple(input_shape)).to(device)
        random_input3 = torch.rand(tuple(input_shape)).to(device)
        yhat = model(random_input1, random_input2, random_input3)
        make_dot(
            yhat,
            params=dict(list(model.named_parameters()))
        ).render(output_path, format=image_format)
