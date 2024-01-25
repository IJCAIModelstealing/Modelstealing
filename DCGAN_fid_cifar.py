from __future__ import print_function
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
import torchvision.utils as vutils
import numpy as np
import sys
# from model import *
file_path = "output_200_grey_cifar.txt"
error_path = "err.txt"# <-comment out to see the output
tmp = sys.stdout
error = sys.stderr # <-comment out to see the output
sys.stdout = open(file_path, "w")   # <-comment out to see the output
sys.stderr = open(error_path, "w")
                  # redirected to the file
                    #returns to normal mode / comment out
cudnn.benchmark = True

# set manual seed to a constant get a consistent output
manualSeed = random.randint(1, 10000)
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# checking the availability of cuda devices
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'


# number of gpu's available
ngpu = 1
# input noise dimension
nz = 100
# number of generator filters
ngf = 64
# number of discriminator filters
ndf = 64

#number of workers
nw = 2

#number of channel
nc = 3

#batch size
batch_size = 64

train_data = dset.CIFAR10(root="./data", download=True,train=True,
                           transform=transforms.Compose([
                               transforms.Resize(ngf),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
test_data = dset.CIFAR10(root="./data", download=True,train=False,
                           transform=transforms.Compose([
                               transforms.Resize(ngf),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
def data_loader(data, batch_size):
    # loading the dataset



    dataloader = torch.utils.data.DataLoader(data, batch_size=len(train_data),
                                             shuffle=True, num_workers=0)
    #

    data = next(iter(dataloader))[0] #
    # data = torch.flatten(data, 1)
    labels = next(iter(dataloader))[1]
    labels = labels.long()
    return data, labels


def get_train_set(train_set, participant_index=0, distribution=None):
    """
    Get the indices for each training batch
    :param participant_index: the index of a particular participant, must be less than the number of participants
    :return: tensor[number_of_batches_allocated, BATCH_SIZE] the indices for each training batch
    """
    if distribution is None:
        batches_per_participant = train_set.size(0) // nw
        # print("batches_per_participant",batches_per_participant)
        lower_bound = participant_index * batches_per_participant
        upper_bound = (participant_index + 1) * batches_per_participant
        dataset = TensorDataset(train_set[lower_bound: upper_bound])
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        data = dataloder2list(dataloader)
        return data

    # #non iid
    # else:
    #     for i in range(nw):
    #





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



# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
            return output

class Generator_MLP(nn.Module):
    def __init__(self, ngpu = 1):
        super(Generator_MLP, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Linear(nz, ngf * 8 * 8 * 8),  # Fully connected layer
            nn.ReLU(True),
            nn.Linear(ngf * 8 * 8 * 8, ngf * 4 * 16 * 16),  # Fully connected layer
            nn.ReLU(True),
            nn.Linear(ngf * 4 * 16 * 16, ngf * 2 * 32 * 32),  # Fully connected layer
            nn.ReLU(True),
            nn.Linear(ngf * 2 * 32 * 32, ngf * 1 * 64 * 64),  # Fully connected layer
            nn.ReLU(True),
            nn.Linear(ngf * 1 * 64 * 64, nc * 64 * 64),  # Fully connected layer
            nn.Tanh()
        )
    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


# class Generator_MLP(nn.Module):
#     def __init__(self, nc, nz, ngf):
#         super(Generator_MLP, self).__init__()
#         self.ngpu = ngpu
#         self.main = nn.Sequential(
#             nn.Linear(nz, ngf * 8 * 4 * 4),  # Fully connected layer
#             nn.ReLU(True),
#             nn.Linear(ngf * 8 * 4 * 4, ngf * 4 * 8 * 8),  # Fully connected layer
#             nn.ReLU(True),
#             nn.Linear(ngf * 4 * 8 * 8, ngf * 2 * 16 * 16),  # Fully connected layer
#             nn.ReLU(True),
#             nn.Linear(ngf * 2 * 16 * 16, ngf * 1 * 32 * 32),  # Fully connected layer
#             nn.ReLU(True),
#             nn.Linear(ngf * 1 * 32 * 32, nc * 32 * 32),  # Fully connected layer
#             nn.Tanh()
#         )
#
#     def forward(self, input):
#         input = input.view(input.size(0), -1)  # Flatten the input tensor
#         if input.is_cuda and self.ngpu > 1:
#             output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
#         else:
#             output = self.main(input)
#         output = output.view(output.size(0), nc, 32, 32) # Reshape to 4D tensor (batch_size, nc, 32, 32)
#         return output

# class StudentGenerator1(nn.Module):
#     def __init__(self):
#         super(StudentGenerator1, self).__init__()
#         self.main = nn.Sequential(
#             # input is Z, going into a convolution
#             nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, bias=False),
#             nn.BatchNorm2d(ngf * 4),
#             nn.ReLU(True),
#             # state size. (ngf*4) x 4 x 4
#             nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf * 2),
#             nn.ReLU(True),
#             # state size. (ngf*2) x 8 x 8
#             nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf),
#             nn.ReLU(True),
#             # state size. (ngf) x 16 x 16
#             nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
#             nn.Tanh(),
#             # state size. (nc) x 32 x 32
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
#             nn.Conv2d(nc, nc, 3, 1, 1),
#             nn.Tanh()
#             # state size. (nc) x 64 x 64
#         )
#
#     def forward(self, input):
#         output = self.main(input)
#         return output
#
# class StudentGenerator(nn.Module):
#     def __init__(self):
#         super(StudentGenerator, self).__init__()
#         self.main = nn.Sequential(
#             # input is Z, going into a convolution
#             nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, bias=False),
#             nn.BatchNorm2d(ngf * 4),
#             nn.ReLU(True),
#             # state size. (ngf*4) x 4 x 4
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
#             nn.Conv2d(ngf * 4, ngf * 2, 3, 1, 1, bias=False),
#             nn.BatchNorm2d(ngf * 2),
#             nn.ReLU(True),
#             # state size. (ngf*2) x 8 x 8
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
#             nn.Conv2d(ngf * 2, ngf, 3, 1, 1, bias=False),
#             nn.BatchNorm2d(ngf),
#             nn.ReLU(True),
#             # state size. (ngf) x 16 x 16
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
#             nn.Conv2d(ngf, nc, 3, 1, 1, bias=False),
#             nn.Tanh(),
#             # state size. (nc) x 32 x 32
#             nn.Upsample(size=(64, 64), mode='bilinear', align_corners=True),
#             nn.Conv2d(nc, nc, 3, 1, 1),
#             nn.Tanh()
#             # state size. (nc) x 64 x 64
#         )
#         #     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
#         #     nn.Conv2d(ngf, ngf, 4, 2, 1, bias=False),
#         #     nn.BatchNorm2d(ngf),
#         #     nn.ReLU(True),
#         #     # state size. (ngf) x 16 x 16
#         #     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
#         #     nn.Conv2d(ngf, nc, 3, 1, 1, bias=False),
#         #     nn.Tanh()
#         #     # state size. (nc) x 64 x 64
#         # )
#
#     def forward(self, input):
#         output = self.main(input)
#         print(output.shape)
#         return output
#
#
#
# class StudentGenerator2(nn.Module):
#     def __init__(self, nz, ngf, nc):
#         super(StudentGenerator2, self).__init__()
#         self.main = nn.Sequential(
#             # input is Z, going into a convolution
#             nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, bias=False),
#             nn.BatchNorm2d(ngf * 4),
#             nn.ReLU(True),
#             # state size. (ngf*4) x 4 x 4
#             nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf * 2),
#             nn.ReLU(True),
#             # state size. (ngf*2) x 8 x 8
#             nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf),
#             nn.ReLU(True),
#             # state size. (ngf) x 16 x 16
#             nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
#             nn.Tanh(),
#             # state size. (nc) x 32 x 32
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
#             nn.Conv2d(nc, nc, 3, 1, 1),
#             nn.Tanh()
#             # state size. (nc) x 64 x 64
#         )
#
#     def forward(self, input):
#         output = self.main(input)
#         return output



class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)

def set_optimizer(net, lr=0.0002, beta1=0.5):
    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(beta1, 0.999))
    return optimizer

def dataloder2list(dataloader):
    data_list = []
    for i, data in enumerate(dataloader, 0):
        data_list.append(data)
    return data_list


def calculate_activation_statistics(images, model, batch_size=128, dims=2048,
                                    cuda=False):
    model.eval()
    act = np.empty((len(images), dims))

    if cuda:
        batch = images.cuda()
    else:
        batch = images
    pred = model(batch)[0]

    # If model output is not scalar, apply global spatial average pooling.
    # This happens if you choose a dimensionality not equal 2048.
    if pred.size(2) != 1 or pred.size(3) != 1:
        pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

    act = pred.cpu().data.numpy().reshape(pred.size(0), -1)

    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

def calculate_fretchet(images_real, images_fake, model):
    mu_1, std_1 = calculate_activation_statistics(images_real, model, cuda=False)
    mu_2, std_2 = calculate_activation_statistics(images_fake, model, cuda=False)

    """get fretched distance"""
    fid_value = calculate_frechet_distance(mu_1, std_1, mu_2, std_2)
    return fid_value



criterion = nn.BCELoss()
criterion_attcker = nn.MSELoss()
L1_loss = nn.L1Loss()
# L2_loss = nn.L


def MSE(img1, img2):
    squared_diff = (img1 - img2) ** 2
    summed = torch.sum(squared_diff)
    num_pix = img1.shape[0] * img1.shape[1]  # img1 and 2 should have same shape
    err = summed / num_pix
    return err

fixed_noise = torch.randn(batch_size, nz, 1, 1, device=device)
real_label = 1
fake_label = 0

niter = 200
g_loss = []
d_loss = []
lamda = 0.5
lr = 0.0002

dataset = 'cifar10'
prefix = 'whitebox'

# def MSE(input, target):
#     return torch.mean((input - target) ** 2)


if __name__ ==  '__main__':
    train_data,train_label = data_loader(train_data,batch_size)
    print(train_data.shape) # dataset shape ([50000, 3, 64, 64])
    participants_train_set  = [get_train_set(train_data, i) for i in range(nw)]
    participant_train_label = [get_train_set(train_label, i) for i in range(nw)]
    # print(len(participants_train_set[0])) # number of batches for each participant
    print(len(participant_train_label[0][0]))  # number of batches for each participant

    output_dir = '{}_{}output_lr{}_lamda{}_image_szie{}'.format(dataset, prefix, lr, lamda, ngf)
    weight_dir = '{}_{}weights_lr{}_lamda{}_image_szie{}'.format(dataset, prefix, lr, lamda, ngf)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)

    #initilize the generator and discriminator
    netG = Generator(ngpu).to(device)
    netG.apply(weights_init)
    netD_list  = [Discriminator(ngpu).to(device) for i in range(nw)]
    for netD in netD_list:
        netD.apply(weights_init)

    # setup optimizer
    optimizerD_list = [set_optimizer(netD_list[i]) for i in range(nw)]
    optimizerG = set_optimizer(netG)
    net_attacker = Generator(ngpu).to(device)
    # net_attacker= Generator(ngpu).to(device)
    net_attacker.apply(weights_init)
    optimizer_attacker = set_optimizer(net_attacker, lr=0.00002)
    print("Attacker's learning rate is 0.00002")
    for epoch in range(niter):
        indice = 0
        # i = 0
        while indice < len(participants_train_set[0])-1:
            print("indice", indice)
            for i, participant_set in enumerate(participants_train_set):
                print(i)
                if i == 0:
                    real_cpu = participant_set[indice][0].to(device)
                    vutils.save_image(real_cpu, '{}/real_samples_begining.png'.format(output_dir), normalize=True)
                    # train with real
                    for attacker_round in range(10):
                        net_attacker.zero_grad()
                        noise = torch.randn(batch_size, nz, 1, 1, device=device)
                        fake = netG(noise)
                        fake_attacker = net_attacker(noise)
                        err_attacker = lamda*criterion_attcker(fake_attacker, real_cpu)+(1-lamda)*criterion_attcker(fake_attacker, fake)
                        err_attacker.backward()
                        optimizer_attacker.step()
                        print("attacker loss: ", err_attacker.item())
                    ############################
                    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                    ###########################
                    # train with real
                netD_list[i].zero_grad()

                batch_size = real_cpu.size(0)
                label = torch.full((batch_size,), real_label, device=device)
                output = netD_list[i](real_cpu)
                errD_real = criterion(output, label.float())
                errD_real.backward()
                D_x = output.mean().item()

                # train with fake
                noise = torch.randn(batch_size, nz, 1, 1, device=device)
                fake = netG(noise)
                label.fill_(fake_label)
                output = netD_list[i](fake.detach())
                errD_fake = criterion(output, label.float())
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                errD = errD_real + errD_fake
                # errD_list.append(errD.item())
                optimizerD_list[i].step()
                print(
                    'for participant %d [%d/%d][%d/%d] Loss_D: %.4f  D(x): %.4f D(G(z)): %.4f ' % (i,epoch,niter,indice,len(participants_train_set[i]),errD.item(),D_x,D_G_z1))


            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
                for l in range(3):
                    fake = netG(noise)
                    netG.zero_grad()
                    label = torch.full((batch_size,), real_label, device=device)  # fake labels are real for generator cost
                    output = [netD_list[i](fake) for i in range(nw)]
                    errG = [criterion(output[i], label.float()) for i in range(nw)]
                    errG = torch.stack(errG).mean()
                    errG.backward()
                    D_G_z2 = output[0].mean().item()
                    optimizerG.step()
                # if indice != len(participants_train_set[i])-1:
                #     indice += 1
                # else:
                #     indice = 0

                    print('for epoch [%d/%d]  Loss_G: %.4f  D_G_z2:  %.4f' % (epoch, niter,   errG.item(),   D_G_z2))
            # save the output
                if indice % 10 == 0:
                    print('saving the output')
                    vutils.save_image(real_cpu, '{}/real_samples.png'.format(output_dir), normalize=True)
                    fake = netG(fixed_noise)
                    steal = net_attacker(fixed_noise)
                    vutils.save_image(fake.detach(), '{}/fake_samples_epoch_%03d.png'.format(output_dir) % (epoch), normalize=True)
                    vutils.save_image(steal.detach(), '{}/steal_samples_epoch_%03d.png'.format(output_dir) % (epoch), normalize=True)

            indice += 1
        # break



        # Check pointing for every epoch
        torch.save(netG.state_dict(), '{}/netG_epoch_%d.pth'.format(weight_dir) % (epoch))
        for i in range(nw):
            torch.save(netD_list[i].state_dict(), '{}/netD_list_%d_epoch_%d.pth'.format(weight_dir) % (i, epoch))
        # torch.save(netD.state_dict(), 'weights/netD_epoch_%d.pth' % (epoch))
        torch.save(net_attacker.state_dict(), '{}/net_attacker_epoch_%d.pth'.format(weight_dir) % (epoch))
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        test_attacker = net_attacker(noise)
        vutils.save_image(test_attacker.detach(), '{}/test_samples_epoch_%03d.png'.format(weight_dir) % (epoch), normalize=True)

    netG.load_state_dict(torch.load('{}/netG_epoch_49.pth'.format(weight_dir)))
    for i in range(nw):
        netD_list[i].load_state_dict(torch.load('{}/netD_list_%d_epoch_49.pth'.format(weight_dir) % i))
    net_attacker.load_state_dict(torch.load('{}/net_attacker_epoch_49.pth'.format(weight_dir)))
    noise = torch.randn(batch_size, nz, 1, 1, device=device)
    fake = netG(noise)
    steal = net_attacker(noise)
    vutils.save_image(fake.detach(), '{}/reload_fake.png'.format(output_dir), normalize=True)
    vutils.save_image(steal.detach(), '{}/reload_steal.png'.format(output_dir), normalize=True)

    test_data, test_label = data_loader(test_data, batch_size)
    print(test_data.shape)  # dataset shape ([50000, 3, 64, 64])
    participants_test_set = [get_train_set(test_data, i) for i in range(nw)]
    print(len(participants_test_set[0]))  # number of batches for each participant
    net_attacker.eval()
    noise = torch.randn(batch_size, nz, 1, 1, device=device)
    test_attacker = net_attacker(noise)
    fretchet_dist = calculate_fretchet(test_data, test_attacker, net_attacker)
    print(fretchet_dist)
    #test_attacker
    vutils.save_image(test_attacker.detach(), '{}/test_attacker.png'.format(output_dir), normalize=True)






# sys.stdout.close()                  # <-comment out to see the output
# sys.stdout = tmp