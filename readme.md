# MoEx

## Code structure
To run the experiments, please run __DCGAN_mode_dataset.py__, replace *mode* with the attack mode to experiment with.
Here, there are three key runable files in the repository, including
* __DCGAN_acc_cifar.py__  # Accuracy extraction on CIFAR10
* __DCGAN_acc_celeba.py__ # Accuracy extraction on Celeba
* __DCGAN_fid_cifar.py__  # Fidelity extraction on CIFAR10
* __DCGAN_fid_celeba.py__ # Fidelity extraction on Celeba

Other files in the repository
* __constants.py__ Experiment constants, contains the default hyperparameter set for each experiment
* __datasets.py__ It is used for read dataset, split dataset into training set and test set for each participant


* __models.py__ The models including target and attack model are wrapped in this file, and also including
  * Basic target and attack models

## Instructions for running the experiments
### 1. Set the experiment parameters
The experiment parameters are defined in __constants.py__. To execute the experiments, please set the parameters in __constants.py__ file. 

### 2. Run the experiment
To run the experiment, please run __DCGAN_*mode*_*dataset*_.py__, replace *mode*  with the attack mode to experiment with. You can use command line to run the experiment, e.g. in a LINUX environment, to execute the *Accuracy extraction* experiment on CIFAR10, please input the following command under the source code path

```python DCGAN_acc_cifar.py```

To execute the *Fidelity extraction* experiment on CIFAR10, please input the following command under the source code path

```python DCGAN_fid_cifar.py```

### 3. Save the experiment results
After the experiment is finished, the experiment results will be saved in the directory, eg. __cifar10_greybox_output_lr0.0002_lamda0.5_image_szie64__. The experiment results include the images generated from target GAN, the real sample use for training and the images generated from *MoEx*.


## Requirements
Recommended to run with conda virtual environment
* Python 3.10.5
* PyTorch 1.13.0
* numpy 1.22.4
* pandas 1.4.2

Required download the dataset：
CelebA: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
