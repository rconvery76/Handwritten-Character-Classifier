import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF
import torch
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

#----- Configurations -----
ROOT = Path('./data')
OUTPUT_DIR = Path('./data/processed')
BATCH_SIZE = 1024
SHARD_SIZE = 10000
NUM_WORKERS = 4




#download the EMIST dataset
train_dataset = torchvision.datasets.EMNIST(
    root='./data',
    split='letters',
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

test_dataset = torchvision.datasets.EMNIST(
    root='./data',
    split='letters',
    train=False,
    download=True,
    transform=transforms.ToTensor()
)


#visualize some samples from the dataset
def show_samples(dataset, num_samples=5):
    plt.figure(figsize=(10, 2))
    for i in range(num_samples):
        image, label = dataset[i]
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(image.squeeze(), cmap='gray')
        plt.title(f'Label: {label}')
        plt.axis('off')


def orient_images(img):
    # Rotate the image by 90 degrees clockwise
    oriented_image = TF.rotate(img, -90)
    # Flip the image horizontally
    oriented_image = TF.hflip(oriented_image)
    return oriented_image

def make_tf(mean, std):
    ops = [
        transforms.Lambda(orient_images),
        transforms.ToTensor()
    ]
    if mean is not None and std is not None:
        ops += [transforms.Normalize([mean], [std])]
    return transforms.Compose(ops)

#remap labels from 1-26 to 0-25
def remap_labels(label):
    return label - 1

#compute the mean and std of the dataset
def compute_mean_std(dataset):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    mean = 0.0
    std = 0.0
    total_samples = 0

    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_samples += batch_samples

    mean /= total_samples
    std /= total_samples
    return mean.item(), std.item()

#process and save the dataset in shards
def process_and_save_dataset(dataset, split_name, mean, std):
    output_dir = OUTPUT_DIR / split_name
    output_dir.mkdir(parents=True, exist_ok=True)

    tf = make_tf(mean, std)
    dataset.transform = tf

    num_shards = (len(dataset) + SHARD_SIZE - 1) // SHARD_SIZE

    for shard_idx in range(num_shards):
        start_idx = shard_idx * SHARD_SIZE
        end_idx = min((shard_idx + 1) * SHARD_SIZE, len(dataset))
        shard_data = []

        for i in range(start_idx, end_idx):
            image, label = dataset[i]
            label = remap_labels(label)
            shard_data.append((image.numpy(), label))

        shard_path = output_dir / f'shard_{shard_idx:03d}.pt'
        torch.save(shard_data, shard_path)
        print(f'Saved {shard_path} with {len(shard_data)} samples.')

def show_image(image, label):
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title(f'Label: {label}')
    plt.axis('off')
    plt.show()

def display_sample_from_shard(shard_path, sample_idx=0):
    shard_data = torch.load(shard_path, weights_only=False, map_location='cpu')
    image, label = shard_data[sample_idx]
    show_image(image, label)





if __name__ == '__main__':
    # Compute mean and std from the training dataset
    train_dataset.transform = make_tf(None, None)
    mean, std = compute_mean_std(train_dataset)
    print(f'Computed Mean: {mean}, Std: {std}')

    # Save the mean and std to a JSON file
    stats = {'mean': mean, 'std': std}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / 'stats.json', 'w') as f:
        json.dump(stats, f)

    # Process and save the training and test datasets
    process_and_save_dataset(train_dataset, 'train', mean, std)
    process_and_save_dataset(test_dataset, 'test', mean, std)

    # Display a few samples from the shards
    # display_sample_from_shard(OUTPUT_DIR / 'train' / 'shard_000.pt', sample_idx=0)
    # display_sample_from_shard(OUTPUT_DIR / 'test' / 'shard_000.pt', sample_idx=0)
    # display_sample_from_shard(OUTPUT_DIR / 'train' / 'shard_000.pt', sample_idx=1)

