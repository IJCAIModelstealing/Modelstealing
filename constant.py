import torch
import random
# set manual seed to a constant get a consistent output
manualSeed = random.randint(1, 10000)
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# checking the availability of cuda devices
device = 'cuda:0'
    # if torch.cuda.is_available() else 'cpu'
dataset_dir = "./data/CelebA/"
model = "DCGAN"
# dataset = "CIFAR"
dataset = "CelebA"
# device = 'cpu'
# number of gpu's available
ngpu = 1
# input noise dimension
nz = 100
# number of generator filters
ngf = 32
# ngf = 64
# number of discriminator filters
ndf = 32
# ndf = 64

#number of workers
nw = 5

#number of channel
nc = 3

#batch size
batch_size = 128
# batch_size = 64
if model == "DCGAN":
    fixed_noise = torch.randn(batch_size, nz, 1, 1, device=device)
elif model == "PGGAN":
    fixed_noise = torch.randn(16, latent_size, 1, 1, device=device)
real_label = 1
fake_label = 0

niter = 200
g_loss = []
d_loss = []
lamda = 0.5 #for blackbox attack
mode = "whitebox"
prefix = "{}_".format(mode)
lr = 0.0002