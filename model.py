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
from constant import *
import torch.nn.functional as F
from torch.autograd import Variable
import copy


class FromRGB(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = EqualizedLR_Conv2d(in_ch, out_ch, kernel_size=(1, 1), stride=(1, 1))
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        return self.relu(x)


class ToRGB(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = EqualizedLR_Conv2d(in_ch, out_ch, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        return self.conv(x)


class G_Block(nn.Module):
    def __init__(self, in_ch, out_ch, initial_block=False):
        super().__init__()
        if initial_block:
            self.upsample = None
            self.conv1 = EqualizedLR_Conv2d(in_ch, out_ch, kernel_size=(4, 4), stride=(1, 1), padding=(3, 3))
        else:
            self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
            self.conv1 = EqualizedLR_Conv2d(in_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = EqualizedLR_Conv2d(out_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu = nn.LeakyReLU(0.2)
        self.pixelwisenorm = Pixel_norm()
        nn.init.normal_(self.conv1.weight)
        nn.init.normal_(self.conv2.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x):

        if self.upsample is not None:
            x = self.upsample(x)
        # x = self.conv1(x*scale1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pixelwisenorm(x)
        # x = self.conv2(x*scale2)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pixelwisenorm(x)
        return x


class D_Block(nn.Module):
    def __init__(self, in_ch, out_ch, initial_block=False):
        super().__init__()

        if initial_block:
            self.minibatchstd = Minibatch_std()
            self.conv1 = EqualizedLR_Conv2d(in_ch + 1, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.conv2 = EqualizedLR_Conv2d(out_ch, out_ch, kernel_size=(4, 4), stride=(1, 1))
            self.outlayer = nn.Sequential(
                nn.Flatten(),
                nn.Linear(out_ch, 1)
            )
        else:
            self.minibatchstd = None
            self.conv1 = EqualizedLR_Conv2d(in_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.conv2 = EqualizedLR_Conv2d(out_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.outlayer = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.relu = nn.LeakyReLU(0.2)
        nn.init.normal_(self.conv1.weight)
        nn.init.normal_(self.conv2.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x):
        if self.minibatchstd is not None:
            x = self.minibatchstd(x)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.outlayer(x)
        return x


class Generator_pggan(nn.Module):
    def __init__(self, latent_size, out_res):
        super().__init__()
        self.depth = 1
        self.alpha = 1
        self.fade_iters = 0
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.current_net = nn.ModuleList([G_Block(latent_size, latent_size, initial_block=True)])
        self.toRGBs = nn.ModuleList([ToRGB(latent_size, 3)])
        # __add_layers(out_res)
        for d in range(2, int(np.log2(out_res))):
            if d < 6:
                ## low res blocks 8x8, 16x16, 32x32 with 512 channels
                in_ch, out_ch = 512, 512
            else:
                ## from 64x64(5th block), the number of channels halved for each block
                in_ch, out_ch = int(512 / 2 ** (d - 6)), int(512 / 2 ** (d - 5))
            self.current_net.append(G_Block(in_ch, out_ch))
            self.toRGBs.append(ToRGB(out_ch, 3))

    def forward(self, x):
        for block in self.current_net[:self.depth - 1]:
            x = block(x)
        out = self.current_net[self.depth - 1](x)
        x_rgb = self.toRGBs[self.depth - 1](out)
        if self.alpha < 1:
            x_old = self.upsample(x)
            old_rgb = self.toRGBs[self.depth - 2](x_old)
            x_rgb = (1 - self.alpha) * old_rgb + self.alpha * x_rgb

            self.alpha += self.fade_iters

        return x_rgb

    def growing_net(self, num_iters):

        self.fade_iters = 1 / num_iters
        self.alpha = 1 / num_iters

        self.depth += 1

class Generator_cifar(nn.Module):
    def __init__(self, ngpu):
        super(Generator_cifar, self).__init__()
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

class Discriminator_cifar(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator_cifar, self).__init__()
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


class Discriminator_pggan(nn.Module):
    def __init__(self, latent_size, out_res):
        super().__init__()
        self.depth = 1
        self.alpha = 1
        self.fade_iters = 0

        self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.current_net = nn.ModuleList([D_Block(latent_size, latent_size, initial_block=True)])
        self.fromRGBs = nn.ModuleList([FromRGB(3, latent_size)])
        for d in range(2, int(np.log2(out_res))):
            if d < 6:
                in_ch, out_ch = 512, 512
            else:
                in_ch, out_ch = int(512 / 2 ** (d - 5)), int(512 / 2 ** (d - 6))
            self.current_net.append(D_Block(in_ch, out_ch))
            self.fromRGBs.append(FromRGB(3, in_ch))

    def forward(self, x_rgb):
        x = self.fromRGBs[self.depth - 1](x_rgb)

        x = self.current_net[self.depth - 1](x)
        if self.alpha < 1:
            x_rgb = self.downsample(x_rgb)
            x_old = self.fromRGBs[self.depth - 2](x_rgb)
            x = (1 - self.alpha) * x_old + self.alpha * x
            self.alpha += self.fade_iters
        for block in reversed(self.current_net[:self.depth - 1]):
            x = block(x)

        return x

    def growing_net(self, num_iters):

        self.fade_iters = 1 / num_iters
        self.alpha = 1 / num_iters

        self.depth += 1


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # state size. (ngf*16) x 4 x 4

            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 8 x 8

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 16 x 16

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 32 x 32

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 64 x 64

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 128 x 128
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

class Generator_MLP(nn.Module):
    def __init__(self, nc, nz, ngf):
        super(Generator_MLP, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(nz, ngf * 8 * 4 * 4),  # Fully connected layer
            nn.ReLU(True),
            nn.Linear(ngf * 8 * 4 * 4, ngf * 4 * 8 * 8),  # Fully connected layer
            nn.ReLU(True),
            nn.Linear(ngf * 4 * 8 * 8, ngf * 2 * 16 * 16),  # Fully connected layer
            nn.ReLU(True),
            nn.Linear(ngf * 2 * 16 * 16, ngf * 1 * 32 * 32),  # Fully connected layer
            nn.ReLU(True),
            nn.Linear(ngf * 1 * 32 * 32, nc * 64 * 64),  # Fully connected layer for 64x64 output
            nn.Tanh()
        )

    def forward(self, input):
        input = input.view(input.size(0), -1)  # Flatten the input tensor
        output = self.main(input)
        output = output.view(output.size(0), nc, 64, 64)  # Reshape to 4D tensor (batch_size, nc, 64, 64)
        return output


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 256 x 256
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 128 x 128

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 64 x 64

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 32 x 32

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 16 x 16

            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*16) x 8 x 8

            nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # state size. 1 x 4 x 4
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)


class Generator_celeba(nn.Module):
    def __init__(self, nc, nz, ngf):
        super().__init__()

        # Input is the latent vector Z.
        self.tconv1 = nn.ConvTranspose2d(nz, ngf * 8,
                                         kernel_size=4, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(ngf * 8)

        # Input Dimension: (ngf*8) x 4 x 4
        self.tconv2 = nn.ConvTranspose2d(ngf * 8, ngf * 4,
                                         4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ngf * 4)

        # Input Dimension: (ngf*4) x 8 x 8
        self.tconv3 = nn.ConvTranspose2d(ngf * 4, ngf * 2,
                                         4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(ngf * 2)

        # Input Dimension: (ngf*2) x 16 x 16
        self.tconv4 = nn.ConvTranspose2d(ngf * 2, ngf,
                                         4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(ngf)

        # Input Dimension: (ngf) * 32 * 32
        self.tconv5 = nn.ConvTranspose2d(ngf, nc,
                                         4, 2, 1, bias=False)
        # Output Dimension: (nc) x 64 x 64

    def forward(self, x):
        x = F.relu(self.bn1(self.tconv1(x)))
        x = F.relu(self.bn2(self.tconv2(x)))
        x = F.relu(self.bn3(self.tconv3(x)))
        x = F.relu(self.bn4(self.tconv4(x)))

        x = torch.tanh(self.tconv5(x))

        return x


class Discriminator_celeba(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator_celeba, self).__init__()
        self.cv1 = nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1, bias=False)  # (3, 64, 64) -> (64, 32, 32)
        self.cv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1)  # (64, 32, 32) -> (128, 16, 16)
        self.bn2 = nn.BatchNorm2d(ndf * 2)  # spatial batch norm is applied on num of channels
        self.cv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1)  # (128, 16, 16) -> (256, 8, 8)
        self.bn3 = nn.BatchNorm2d(ndf * 4)
        self.cv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)  # (256, 8, 8) -> (512, 4, 4)
        self.bn4 = nn.BatchNorm2d(ndf * 8)
        self.cv5 = nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)  # (512, 4, 4) -> (1, 1, 1)

    def forward(self, x):
        x = F.leaky_relu(self.cv1(x))
        x = F.leaky_relu(self.bn2(self.cv2(x)), 0.2, True)
        x = F.leaky_relu(self.bn3(self.cv3(x)), 0.2, True)
        x = F.leaky_relu(self.bn4(self.cv4(x)), 0.2, True)
        x = torch.sigmoid(self.cv5(x))
        return x.view(-1, 1).squeeze(1)


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection if dimensions change
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = torch.relu(out)
        print("Block Output Size:", out.size())
        return out

class ResNetGenerator(nn.Module):
    def __init__(self, z_dim, image_channels=3, ngf=64):
        super(ResNetGenerator, self).__init__()
        self.fc = nn.Linear(z_dim, 8 * ngf * 8 * 8)
        self.block1 = ResNetBlock(8 * ngf, 4 * ngf, stride=2)
        self.block2 = ResNetBlock(4 * ngf, 2 * ngf, stride=2)
        self.block3 = ResNetBlock(2 * ngf, ngf, stride=2)
        self.block4 = ResNetBlock(ngf, ngf, stride=2)
        self.bn = nn.BatchNorm2d(ngf)
        self.tconv_out = nn.ConvTranspose2d(ngf, image_channels, kernel_size=4, stride=2, padding=1, bias=False)

    def forward(self, x):
        x = self.fc(x)
        x = x = x.view(x.size(0), -1, 8, 8)
        print("After FC View Size:", x.size())
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = torch.relu(self.bn(x))
        x = torch.tanh(self.tconv_out(x))
        print("Final Output Size:", x.size())
        return x