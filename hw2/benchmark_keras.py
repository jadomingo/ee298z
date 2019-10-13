#!/usr/bin/env python3
import numpy as np
import keras
from keras import backend as K

from classifier_keras import model as classifier
from structural_similarity import structural_similarity as ssim


class IdentityModel(keras.Model):
    """Model which simply returns the input"""

    def __init__(self):
        super().__init__()
        self.identity = keras.layers.Lambda(lambda x: x)

    def call(self, x):
        return self.identity(x)


def _preprocess_for_classifier(x):
    return (x - 0.1307) / 0.3081


def test_model(model, x_test, y_test, batch_size=100):
    """Run the benchmarks for the given model

    :param model:
    :param x_test: MNIST images scaled to [0, 1]
    :param y_test: MNIST labels, raw values, not one-hot vectors
    :param batch_size: batch size to use for evaluation
    :return: None
    """
    rng = np.random.RandomState(0)

    channel_dim = 1 if K.image_data_format() == 'channels_first' else -1
    classifier.load_weights('mnist_cnn.h5', by_name=True)

    baseline_score = 0
    correct_score = 0
    ssim_score = 0

    N = len(x_test)
    assert N % batch_size == 0, 'N should be divisible by batch_size'
    num_batches = N // batch_size
    for i in range(num_batches):
        # Shape: (B, 28, 28, 1) or (B, 1, 28, 28) if channels_first
        imgs_orig = np.expand_dims(x_test[batch_size*i:batch_size*(i + 1)], channel_dim)
        # Shape: (B,)
        labels = y_test[batch_size*i:batch_size*(i + 1)]

        # Create corruption masks
        masks = []
        for _ in range(batch_size):
            # Choose square size
            s = rng.randint(7, 15)
            # Choose top-left corner position
            x = rng.randint(0, 29 - s)
            y = rng.randint(0, 29 - s)
            mask = np.zeros(imgs_orig.shape[1:], dtype=np.bool)
            # Set mask area
            mask[y:y + s, x:x + s, :] = True
            masks.append(mask)
        masks = np.stack(masks)

        imgs_corrupted = imgs_orig.copy()
        # Draw squares
        imgs_corrupted[masks] = 1.

        # Generate restored images
        imgs_restored = model.predict_on_batch(imgs_corrupted)

        predicted_labels_orig = classifier.predict_on_batch(_preprocess_for_classifier(imgs_orig)).argmax(axis=-1)
        predicted_labels_restored = classifier.predict_on_batch(_preprocess_for_classifier(imgs_restored)).argmax(axis=-1)
        # Calculate classifier score
        baseline = labels == predicted_labels_orig
        correct = baseline == (labels == predicted_labels_restored)
        baseline_score += int(baseline.sum())
        correct_score += int(correct.sum())

        ssim_score += ssim(imgs_orig[~masks].squeeze(), imgs_restored[~masks].squeeze())

    classifier_score = correct_score / baseline_score
    ssim_score /= num_batches

    print('Classifier score: {:.2f}\nSSIM score: {:.2f}'.format(100 * classifier_score, 100 * ssim_score))
