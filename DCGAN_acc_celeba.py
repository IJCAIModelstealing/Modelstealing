from __future__ import print_function
import os
import torch.nn.parallel
import torch.utils.data
import sys
from model import *
from constant import *
from utils.datasets import *

file_path = "{}_out_{}_{}_acc.txt".format(prefix, dataset,ngf)
error_path = "{}_err_{}_{}_acc.txt".format(prefix, dataset,ngf) # <-comment out to see the output
tmp = sys.stdout
error = sys.stderr # <-comment out to see the output
sys.stdout = open(file_path, "w")   # <-comment out to see the output
sys.stderr = open(error_path, "w")
                  # redirected to the file
                    #returns to normal mode / comment out
cudnn.benchmark = True



# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)




def set_optimizer(net, lr=0.0002, beta1=0.5):
    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(beta1, 0.999))
    return optimizer




def calculate_activation_statistics(images, model, batch_size=128, dims=2048,
                                    cuda=True):
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

def MSE(img1, img2):
    squared_diff = (img1 - img2) ** 2
    summed = torch.sum(squared_diff)
    num_pix = img1.shape[0] * img1.shape[1]  # img1 and 2 should have same shape
    err = summed / num_pix
    return err

def noise_generator(batch_size, nz, device, compare_noise, noise_set_size=100):
    noise_set = []
    score_set = []
    for i in range(noise_set_size):
        generator_noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake_attacker = net_attacker(generator_noise)
        score = MSE(fake_attacker, compare_noise).cpu().detach().numpy()
        noise_set.append(generator_noise)
        score_set.append(score)
    print(score_set)
    best_score = score_set[np.argmin(score_set)]    # best score is the smallest one
    print(best_score)
    best_noise = noise_set[np.argmin(score_set)]  # best noise is the one that generate the smallest score
    return best_noise

def calculate_distance(net1,net2, device):
    generator_noise = torch.randn(batch_size, nz, 1, 1, device=device)
    output1 = net1(generator_noise)
    output2 = net2(generator_noise)
    score = MSE(output1,output2).item()
    return score


criterion = nn.BCELoss()
criterion_attcker = nn.MSELoss()
L1_loss = nn.L1Loss()
# L2_loss = nn.L




if __name__ ==  '__main__':
    train_data,train_label,test_data, test_label= dataset_load(dataset)
    print(train_data.shape)
    # train_data,train_label = data_loader(train_data,batch_size)
    print(train_data.shape) # dataset shape ([50000, 3, 64, 64])
    participants_train_set  = [get_train_set(train_data, i) for i in range(nw)]
    participant_train_label = [get_train_set(train_label, i) for i in range(nw)]
    # print(len(participants_train_set[0])) # number of batches for each participant
    print(len(participant_train_label[0][0]))  # number of batches for each participant

    #initilize the generator and discriminator
    if dataset == 'CelebA':
        netG = Generator_celeba(nc,nz,ngf).to(device)
    elif dataset == 'CIFAR':
        netG = Generator_cifar(ngpu).to(device)
    print(netG)
    netG.apply(weights_init)
    if dataset == 'CelebA':
        netD_list  = [Discriminator_celeba(nc, ndf).to(device) for i in range(nw)]
    elif dataset == 'CIFAR':
        netD_list  = [Discriminator_cifar(ngf).to(device) for i in range(nw)]
    for netD in netD_list:
        netD.apply(weights_init)

    # setup optimizer
    optimizerD_list = [set_optimizer(netD_list[i]) for i in range(nw)]
    optimizerG = set_optimizer(netG)
    if dataset == 'CelebA':
        net_attacker = Generator_celeba(nc, nz, ngf).to(device)
    elif dataset == 'CIFAR':
        net_attacker = Generator_cifar(ngpu).to(device)
    net_attacker.apply(weights_init)
    optimizer_attacker = set_optimizer(net_attacker, lr=lr)
    print("Attacker's learning rate is 0.00002")
    attacker_discriminator = Discriminator_celeba(nc, ndf).to(device)
    attacker_discriminator.apply(weights_init)
    optimizer_attacker_discriminator = set_optimizer(attacker_discriminator, lr=lr)

    output_dir = '{}_{}output_lr{}_lamda{}_image_size{}_acc'.format(dataset, prefix, lr, lamda,ngf)
    weight_dir = '{}_{}weights_lr{}_lamda{}_image_size{}_acc'.format(dataset, prefix, lr, lamda,ngf)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)

    print(len(participants_train_set[0]))
    for epoch in range(niter):
        indice = 0
        # i = 0
        while indice < len(participants_train_set[0]) - 1:
            print("indice", indice)
            for i, participant_set in enumerate(participants_train_set):
                print(i)
                if i == 0:
                    real_cpu = participant_set[indice][0].to(device)
                    vutils.save_image(real_cpu, '{}/real_samples_begining.png'.format(output_dir), normalize=True)
                    noise = torch.randn(128, nz, 1, 1, device=device)
                    # train with real
                    for attacker_round in range(20):
                        attack_step = 0
                        fake = netG(noise)
                        net_attacker.zero_grad()
                        err_fake = attacker_discriminator(real_cpu).mean() - attacker_discriminator(fake).mean()
                        print("the result for real: ", attacker_discriminator(real_cpu).mean().item())
                        print("the result for fake: ", attacker_discriminator(fake).mean().item())
                        print("the result for discriminator loss: ", err_fake.item())
                        if err_fake.item() > 0 and attack_step < 10:
                            noise = torch.randn(128, nz, 1, 1, device=device)
                            fake = netG(noise)
                            fake_attacker = net_attacker(noise)
                            err_attacker = lamda * criterion_attcker(fake_attacker, real_cpu) + (
                                        1 - lamda) * criterion_attcker(fake_attacker, fake)
                            err_attacker.backward()
                            optimizer_attacker.step()
                            print("attacker loss: ", err_attacker.item())
                            attack_step += 1
                    ############################
                    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                    ###########################
                    # train with real
                netD_list[i].zero_grad()
                attacker_discriminator.zero_grad()

                batch_size = real_cpu.size(0)
                label = torch.full((batch_size,), real_label, device=device)
                output = netD_list[i](real_cpu)
                output_attacker = attacker_discriminator(real_cpu)
                errD_real = criterion(output, label.float())
                err_attack_real = criterion(output_attacker, label.float())
                errD_real.backward()
                err_attack_real.backward()
                D_x = output.mean().item()
                attack_disc_x = output_attacker.mean().item()
                # train with fake
                noise = torch.randn(batch_size, nz, 1, 1, device=device)
                fake = netG(noise)
                label.fill_(fake_label)
                output = netD_list[i](fake.detach())
                output_attacker = attacker_discriminator(fake.detach())
                errD_fake = criterion(output, label.float())
                err_attack_fake = criterion(output_attacker, label.float())

                errD_fake.backward()
                err_attack_fake.backward()
                D_G_z1 = output.mean().item()
                attack_disc_G_z1 = output_attacker.mean().item()
                errD = errD_real + errD_fake
                err_attack = err_attack_real + err_attack_fake
                optimizer_attacker_discriminator.step()
                # errD_list.append(errD.item())
                optimizerD_list[i].step()
                print(
                    'for participant %d [%d/%d][%d/%d] Loss_D: %.4f  D(x): %.4f D(G(z)): %.4f ' % (
                        i, epoch, niter, indice, len(participants_train_set[i]), errD.item(), D_x, D_G_z1))

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                for l in range(3):
                    fake = netG(noise)
                    netG.zero_grad()
                    label = torch.full((batch_size,), real_label,
                                       device=device)  # fake labels are real for generator cost
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

                    print('for epoch [%d/%d]  Loss_G: %.4f  D_G_z2:  %.4f' % (epoch, niter, errG.item(), D_G_z2))
                # save the output
                if indice % 10 == 0:
                    print('saving the output')
                    vutils.save_image(real_cpu, '{}/real_samples.png'.format(output_dir), normalize=True)
                    fake = netG(fixed_noise)
                    steal = net_attacker(fixed_noise)
                    vutils.save_image(fake.detach(), '{}/fake_samples_epoch_%03d.png'.format(output_dir) % (epoch),
                                      normalize=True)
                    vutils.save_image(steal.detach(), '{}/steal_samples_epoch_%03d.png'.format(output_dir) % (epoch),
                                      normalize=True)

            indice += 1



        # Check pointing for every epoch
        if epoch % 20 == 0:
            torch.save(netG.state_dict(), '{}/netG_epoch_%d.pth'.format(weight_dir) % (epoch))
            for i in range(nw):
                torch.save(netD_list[i].state_dict(), '{}/netD_list_%d_epoch_%d.pth'.format(weight_dir) % (i, epoch))
        # torch.save(netD.state_dict(), 'weights/netD_epoch_%d.pth' % (epoch))
            torch.save(net_attacker.state_dict(), '{}/net_attacker_epoch_%d.pth'.format(weight_dir)  % (epoch))
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        test_attacker = net_attacker(noise)
        vutils.save_image(test_attacker.detach(), '{}/test_samples_epoch_%03d.png'.format(output_dir) % (epoch), normalize=True)

    netG.load_state_dict(torch.load('{}/netG_epoch_49.pth'.format(weight_dir)))
    for i in range(nw):
        netD_list[i].load_state_dict(torch.load('{}/netD_list_%d_epoch_49.pth'.format(weight_dir) % i))
    net_attacker.load_state_dict(torch.load('{}/net_attacker_epoch_49.pth'.format(weight_dir)))
    noise = torch.randn(batch_size, nz, 1, 1, device=device)
    fake = netG(noise)
    steal = net_attacker(noise)
    vutils.save_image(fake.detach(), '{}/reload_fake.png'.format(output_dir), normalize=True)
    vutils.save_image(steal.detach(), '{}/reload_steal.png'.format(output_dir), normalize=True)