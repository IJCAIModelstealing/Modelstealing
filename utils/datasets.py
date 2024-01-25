from __future__ import print_function
import os
import torch.nn.parallel
import torch.utils.data
import sys
from model import *
from constant import *
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import CelebA
from torchvision.datasets.utils import download_and_extract_archive
from PIL import Image
from torchvision.datasets import ImageFolder
import os
import gdown
from zipfile import ZipFile


def dataset_load(dataset:str,batch_size:int = 128):
    if dataset == "CIFAR":
        train_data = dset.CIFAR10(root="./data", download=True,train=True,
                               transform=transforms.Compose([
                                   transforms.Resize(64),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
        test_data = dset.CIFAR10(root="./data", download=True,train=False,
                               transform=transforms.Compose([
                                   transforms.Resize(64),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),


                               ]))


        # Create PyTorch DataLoaders for training and test sets
        train_data, train_label = data_loader(train_data)
        test_data, test_label = data_loader(test_data)
        print("train_data", train_data.shape)
        return train_data,train_label,test_data,test_label

    elif dataset == "CelebA":
    # Define the CelebA dataset root directory
    #     os.makedirs('./data/CelebA')
        celeba_root = './data/CelebA/'  # Change this to your desired directory


        # Download and extract CelebA dataset
        if not os.path.exists(os.path.join(celeba_root, 'img_align_celeba')):
            url = "https://drive.google.com/uc?id=1O7m1010EJjLE5QxLZiM9Fpjs7Oj6e684"
            output = "./data/CelebA/celeba.zip"
            gdown.download(url, output, quiet=True)

            with ZipFile("./data/CelebA/celeba.zip", "r") as zipobj:
                zipobj.extractall("./data/CelebA")

    # Define image transformations
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Assuming 3 channels (RGB)
        ])

        # Create CelebA dataset
        celeba_dataset = ImageFolder(celeba_root, transform=transform)


        # celeba_loader = DataLoader(celeba_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        train_size = int(0.5 * len(celeba_dataset))
        test_size = len(celeba_dataset) - train_size
        print("train_size",train_size)
        print("test_size",test_size)
        train_dataset, test_dataset = random_split(celeba_dataset, [train_size, test_size])
        print("train_data",train_dataset[0][0].shape)
        print("test_data", test_dataset[0][0].shape)
        # Create PyTorch DataLoaders for training and test sets

        train_data,train_label = data_loader(train_dataset)
        test_data,test_label = data_loader(test_dataset)
        print("train_data",train_data.shape)
        # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        #
        # # Example of using the DataLoader
        # for batch in train_loader:
        #     # Your GAN training code for the training set
        #     print(f"Training Batch shape: {batch[0].shape}")
        #     break  # Break after one iteration for demonstration purposes
        #
        # for batch in test_loader:
        #     # Your GAN evaluation code for the test set
        #     print(f"Test Batch shape: {batch[0].shape}")
        #     break  # Break after one iteration for demonstration purposes
        #
        return train_data,train_label,test_data,test_label




def data_loader(data):
    # loading the dataset

    dataloader = DataLoader(data, batch_size=len(data), shuffle=True, num_workers=4)
    data = next(iter(dataloader))[0] #
    # data = torch.flatten(data, 1)
    labels = next(iter(dataloader))[1]
    labels = labels.long()
    return data, labels


def get_train_set(train_set, participant_index=0):
    """
    Get the indices for each training batch
    :param participant_index: the index of a particular participant, must be less than the number of participants
    :return: tensor[number_of_batches_allocated, BATCH_SIZE] the indices for each training batch
    """
    batches_per_participant = train_set.size(0) // nw
    # print("batches_per_participant",batches_per_participant)
    lower_bound = participant_index * batches_per_participant
    upper_bound = (participant_index + 1) * batches_per_participant
    dataset = TensorDataset(train_set[lower_bound: upper_bound])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    data = dataloder2list(dataloader)
    return data


def get_test_set(data, participant_index=0, distribution=None):
    """
    Get the indices for each test batch
    :param participant_index: the index of a particular participant, must be less than the number of participants
    :return: tensor[number_of_batches_allocated, BATCH_SIZE] the indices for each test batch
    """
    if distribution is None:
        batches_per_participant = data.test_set.size(0) // nw
        lower_bound = participant_index * batches_per_participant
        upper_bound = (participant_index + 1) * batches_per_participant
        return data.test_set[lower_bound: upper_bound]

def dataloder2list(dataloader):
    data_list = []
    for i, data in enumerate(dataloader, 0):
        data_list.append(data)
    return data_list