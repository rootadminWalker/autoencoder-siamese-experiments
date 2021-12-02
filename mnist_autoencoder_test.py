import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.utils import plot_model

from models import SimpleAutoEncoder


def main(args):
    optimizers_map = {
        'adam': Adam,
        'SGD': SGD
    }

    output_base_path = args['output_path']
    model_checkpoints_path = os.path.join(output_base_path, 'model_checkpoints')
    model_structures_path = os.path.join(output_base_path, 'model_structures')
    info_file_path = os.path.join(output_base_path, 'info.json')

    if not os.path.exists(output_base_path):
        os.mkdir(output_base_path)
        os.mkdir(model_checkpoints_path)
        os.mkdir(model_structures_path)

    encoder, decoder, autoencoder = SimpleAutoEncoder.build(input_shape=(28, 28, 1), latent_dim=args['latent_dim'])

    plot_model(encoder, show_shapes=True, to_file=os.path.join(model_structures_path, 'encoder.png'))
    plot_model(decoder, show_shapes=True, to_file=os.path.join(model_structures_path, 'decoder.png'))
    plot_model(autoencoder, show_shapes=True, to_file=os.path.join(model_structures_path, 'autoencoder.png'))

    encoder_output_path = os.path.join(output_base_path, 'encoder.h5')
    decoder_output_path = os.path.join(output_base_path, 'decoder.h5')
    autoencoder_output_path = os.path.join(output_base_path, 'autoencoder.h5')

    EPOCHS = args['epochs']
    BATCH_SIZE = args['batch_size']
    LOSS = args['loss']
    OPT = args['optimizer']

    ((trainX, _), (_, _)) = mnist.load_data()

    trainX = np.expand_dims(trainX, axis=-1)
    trainX = trainX.astype("float32") / 255.0

    (trainX, validX, _, _) = train_test_split(trainX, trainX, test_size=0.25, random_state=42)

    tensorboard = TensorBoard(
        log_dir=os.path.join(output_base_path, 'tensorboard_logs'),
        update_freq='epoch',
    )

    checkpoint_fname = 'model_checkpoints/checkpoint-ep{epoch:02d}-vl{val_loss:.4f}.h5'
    model_checkpoint = ModelCheckpoint(
        filepath=os.path.join(output_base_path, checkpoint_fname),
        monitor='val_loss',
        model='min',
        save_best_only=True,
        verbose=1
    )

    opt = optimizers_map[OPT](lr=args['lr'])
    autoencoder.compile(loss=LOSS, optimizer=opt, metrics=['accuracy'])

    history = autoencoder.fit(
        trainX, trainX,
        validation_data=(validX, validX),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[tensorboard, model_checkpoint]
    )

    encoder.save(encoder_output_path)
    decoder.save(decoder_output_path)
    autoencoder.save(autoencoder_output_path)

    N = np.arange(0, EPOCHS)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, history.history["loss"], label="train_loss")
    plt.plot(N, history.history["val_loss"], label="val_loss")
    plt.plot(N, history.history["accuracy"], label="train_acc")
    plt.plot(N, history.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")

    with open(info_file_path, 'w+') as f:
        json.dump({
            'trained_epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'loss_func': LOSS,
            'optimizer': OPT
        }, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output-path', type=str, required=True,
                        help="Output path for the training result, includes tensorboard logs and model checkpoints")
    parser.add_argument('-e', '--epochs', type=int, required=True,
                        help="Epochs of training")
    parser.add_argument('-b', '--batch-size', type=int, default=32,
                        help="Batch size for training")
    parser.add_argument('-l', '--loss', type=str, default='mse',
                        help="Loss function for training")
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='Optimizer for the training')
    parser.add_argument('-lr', type=float, default=1e-3,
                        help='Learning rate for the training')
    parser.add_argument('--latent-dim', type=int, default=16,
                        help='Num of latent space representation dims')

    args = vars(parser.parse_args())
    main(args)
