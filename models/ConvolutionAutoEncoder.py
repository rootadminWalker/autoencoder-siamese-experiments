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

from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Model


class ConvolutionAutoEncoder:
    @staticmethod
    def build(input_shape, latent_dim):
        encoder_in = Input(shape=input_shape)
        x = encoder_in
        for _ in range(2):
            x = Conv2D(
                filters=32,
                kernel_size=(3, 3),
                # strides=2,
                activation='relu'
            )(x)
        for _ in range(2):
            x = Conv2D(
                filters=64,
                kernel_size=(1, 1),
                activation='relu'
            )(x)

        x = Flatten()(x)
        latent = Dense(units=latent_dim)(x)
        encoder = Model(encoder_in, latent, name="encoder")

        # Decoder
        decoder_in = Input(shape=(latent_dim,))
        y = decoder_in
        y = Dense(
            units=36864,
        )(y)
        y = Reshape(target_shape=(24, 24, 64))(y)
        for _ in range(2):
            y = Conv2DTranspose(
                filters=64,
                kernel_size=(1, 1),
                strides=(1, 1),
                activation='relu'
            )(y)

        for _ in range(2):
            y = Conv2DTranspose(
                filters=32,
                kernel_size=(3, 3),
                # strides=2,
                activation='relu'
            )(y)
        decoder_out = Conv2DTranspose(
            filters=1,
            kernel_size=(1, 1),
            activation='sigmoid'
        )(y)

        decoder = Model(decoder_in, decoder_out, name="decoder")

        autoencoder = Model(encoder_in, decoder(encoder(encoder_in)), name="autoencoder")
        return encoder, decoder, autoencoder
        # return encoder


if __name__ == '__main__':
    encoder, decoder, autoencoder = ConvolutionAutoEncoder.build(input_shape=(28, 28, 1), latent_dim=16)
    print(encoder.summary())
    print(decoder.summary())
