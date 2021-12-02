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

from tensorflow.keras.layers import Input, Dense, Reshape, Flatten
from tensorflow.keras.models import Model


class SimpleAutoEncoder:
    @staticmethod
    def build(input_shape, latent_dim):
        encoder_in = Input(shape=input_shape)

        # Encoder
        x = encoder_in
        x = Flatten()(x)
        x = Dense(
            units=512,
            activation='relu'
        )(x)
        latent = Dense(units=latent_dim)(x)
        encoder = Model(encoder_in, latent, name="encoder")

        # Decoder
        decoder_in = Input(shape=(latent_dim,))
        y = decoder_in
        y = Dense(
            units=512,
            activation='relu'
        )(y)
        y = Dense(
            units=784,
            activation='sigmoid'
        )(y)
        decoder_out = Reshape(target_shape=input_shape)(y)
        decoder = Model(decoder_in, decoder_out, name="decoder")

        autoencoder = Model(encoder_in, decoder(encoder(encoder_in)), name="autoencoder")
        return encoder, decoder, autoencoder
