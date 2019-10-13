#!/usr/bin/env python3
import numpy as np
import torch
from torch import nn

from classifier_torch import Net
from structural_similarity import structural_similarity as ssim


class IdentityModel(nn.Module):
    """Model which simply returns the input"""

    def __init__(self):
        super().__init__()
        self.unused = nn.Linear(1, 1)

    def forward(self, x):
        return x


def _preprocess_for_classifier(x):
    return (x - 0.1307) / 0.3081


def test_model(model, dataset, batch_size=100):
    """Run the benchmarks for the given model

    :param model:
    :param dataset: MNIST dataset with transform=ToTensor() only
    :param batch_size: batch size to use for evaluation
    :return: None
    """
    rng = np.random.RandomState(0)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    classifier = Net()
    classifier.load_state_dict(torch.load('mnist_cnn.pt', map_location=device))
    # Switch to evaluation mode
    classifier.eval()

    baseline_score = 0
    correct_score = 0
    ssim_score = 0

    N = len(dataset)
    assert N % batch_size == 0, 'N should be divisible by batch_size'
    num_batches = N // batch_size
    for i in range(num_batches):
        imgs_orig = []
        labels = []
        # Create corruption masks
        masks = []
        for j in range(batch_size*i, batch_size*(i + 1)):
            img, label = dataset[j]
            imgs_orig.append(img)
            labels.append(label)
            # Choose square size
            s = rng.randint(7, 15)
            # Choose top-left corner position
            x = rng.randint(0, 29 - s)
            y = rng.randint(0, 29 - s)
            mask = torch.zeros((1, 28, 28), dtype=torch.uint8)
            # Set mask area
            mask[:, y:y + s, x:x + s] = 1
            masks.append(mask)
        imgs_orig = torch.stack(imgs_orig).to(device)
        labels = torch.as_tensor(labels, device=device)
        masks = torch.stack(masks).to(device)

        imgs_corrupted = imgs_orig.clone()
        # Draw squares
        imgs_corrupted[masks] = 1.

        # Generate restored images
        model_device = next(model.parameters()).device
        imgs_restored = model(imgs_corrupted.to(model_device)).to(device)

        predicted_labels_orig = classifier(_preprocess_for_classifier(imgs_orig)).argmax(dim=-1)
        predicted_labels_restored = classifier(_preprocess_for_classifier(imgs_restored)).argmax(dim=-1)
        # Calculate classifier score
        baseline = labels == predicted_labels_orig
        correct = baseline == (labels == predicted_labels_restored)
        baseline_score += int(baseline.sum())
        correct_score += int(correct.sum())

        ssim_score += ssim(imgs_orig[~masks].squeeze().numpy(), imgs_restored[~masks].squeeze().numpy())

    classifier_score = correct_score / baseline_score
    ssim_score /= num_batches

    print('Classifier score: {:.2f}\nSSIM score: {:.2f}'.format(100 * classifier_score, 100 * ssim_score))
