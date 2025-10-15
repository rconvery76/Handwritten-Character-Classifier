import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

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
    plt.show()


show_samples(train_dataset)
show_samples(test_dataset)


