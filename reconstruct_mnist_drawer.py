from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
import cv2 as cv
import numpy as np
from datetime import datetime

# autoencoder = load_model('outputs/mnist_autoencoder_outputs/latent_dim_2/autoencoder.h5')
encoder = load_model('outputs/mnist_autoencoder_outputs/Conv/conv3x3&1x1_latent_dim_16/encoder.h5')
decoder = load_model('outputs/mnist_autoencoder_outputs/Conv/conv3x3&1x1_latent_dim_16/decoder.h5')

(_, _), (testX, _) = mnist.load_data()
testX = np.expand_dims(testX, axis=-1)
testX = testX.astype("float32") / 255.0

for test_img in testX:
    original = cv.resize(test_img, (300, 300))

    latent = encoder.predict(np.array([test_img]))
    print(latent[0])
    reconstructed = decoder.predict(latent)[0]

    # reconstructed = autoencoder.predict(np.array([test_img]))[0]
    reconstructed = cv.resize(reconstructed, (300, 300))

    merged = np.hstack((original, reconstructed))
    cv.imshow('original&reconstructed', merged)
    key = cv.waitKey(0) & 0xFF
    if key in [27, ord('q')]:
        break
    elif key == ord('s'):
        cv.imwrite(f'/tmp/ok{datetime.now()}.jpeg', merged * 255)
    elif key == ord('g'):
        cv.imwrite(f'/tmp/glitched{datetime.now()}.jpeg', merged * 255)

cv.destroyAllWindows()
