import numpy as np
import cv2
import os
import json
import torch
import multiprocessing

import argparse

from torch import optim, nn
from torch.utils.data import Dataset, DataLoader
import torchvision.utils as vutils

import matplotlib.pyplot as plt

from PIL import Image

class EmojiDataset(Dataset):

    def __init__(self):
        datapath = os.getcwd() + '/drive/My Drive/DGM/data/'
        
        dirs = [dirname for _, dirname, _ in os.walk(datapath)][0]
        files = []
        with open(datapath+'emoji_pretty.json') as json_file:
            data = json.load(json_file)
            for entry in data:
                if entry['category'] == 'Smileys & Emotion': # or entry['category'] == 'People & Body':
                    files.append(entry['image'])
        images = []
        for file in files:
            for dir in dirs:
                # print('read', datapath + dir + '/' + file)
                im = np.array(cv2.imread(datapath + dir + '/' + file, cv2.IMREAD_UNCHANGED))
                try:
                    im_rgba = im / 255.0
                    im_rgba = np.moveaxis(im_rgba, -1, 0)
                    im_rgba[[0, 2], :, :] = im_rgba[[2, 0], :, :]
                    if im_rgba.shape[0] == 3:
                        continue
                    images.append(im_rgba)
                except:
                    continue
                
                # import matplotlib.pyplot as plt
                # im_rgba = np.moveaxis(im_rgba, 0, 2)
                # im_rgba[:, :, [0, 2]] = im_rgba[:, :, [2, 0]]
                # plt.imsave('test.png', im_rgba)
                

        self.data = np.array(images)
        print(self.data.shape)
        
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]).float(), torch.zeros(1)


class Generator(nn.Module):
    '''
    Generator class
    :param nc: (int) number of channels in training image (in)
    :param nz: (int) size of latent vector z (out)    
    :param ngf: (int) size of feature maps in generator
    :return: (Generator) generator
    '''

    def __init__(self, nc, nz, ngf):
        super(Generator, self).__init__()

        self.conv = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.conv(input)


class Discriminator(nn.Module):
    '''
    Discriminator class
    :param nc: (int) number of channels in training image (in)
    :param ngf: (int) size of feature maps in discriminator
    :return: (Discriminator) discriminator
    '''

    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        
        self.conv = nn.Sequential(
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
        return self.conv(input)

def wasserstein_loss(y_true, y_pred):
	    return torch.mean(torch.mul(y_true, y_pred))

class Training():

    def __init__(self, args):
        
        self.args = args

        self.latent_vector_size = args.latent_vector_size
        self.image_channels = 4

        self.generator = Generator(self.image_channels, self.latent_vector_size, 64)
        self.discriminator = Discriminator(self.image_channels, 64)

        self.generator.apply(self.weights_init)
        self.discriminator.apply(self.weights_init)
        
        lr = args.lr
        beta1 = args.beta1
        beta2 = args.beta2

        self.optim_G = optim.Adam(self.generator.parameters(),lr=lr, betas=(beta1, beta2))
        self.optim_D = optim.Adam(self.discriminator.parameters(),lr=lr, betas=(beta1, beta2))
        
        if args.loss == 'wasserstein':
            self.loss = wasserstein_loss
        else:
            self.loss = nn.BCELoss()
        
        
	
        self.epochs = args.epochs
        self.batch_size = args.batch_size

        self.img_shape = (64, 64)
        
        num_workers = multiprocessing.cpu_count()
        
        print('num_workers', num_workers)
        
        self.dataloader = torch.utils.data.DataLoader(
            EmojiDataset(),
            batch_size=self.batch_size,
            num_workers=num_workers,
            shuffle=True
        )
        
        self.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        print('device',self.device)
	
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def train(self):
        
        # send models to CPU/GPU
        self.discriminator = self.discriminator.to(self.device)
        self.generator = self.generator.to(self.device)

        # establish real and fake interpretation
        real_label = 1
        fake_label = 0

        # fixed noise for visualization
        fixed_noise = torch.randn(64, self.latent_vector_size, 1, 1, device=self.device)

        # tracking
        img_list = []
        G_losses = []
        D_losses = []
        iters = 0

        for epoch in range(self.epochs):
            for i, data in enumerate(self.dataloader, 0):

            # 1. update discriminator
                
                # 1.1 train with all-real batch
                self.discriminator.zero_grad()
                # format batch
                real = data[0].to(self.device)
                batch_size = real.size(0)
                label = torch.full((batch_size,), real_label, device=self.device)
                # forward pass real through discriminator
                output = self.discriminator(real).view(-1)
                # calc loss all-real batch
                errD_real = self.loss(output, label)
                # calc gradients for discriminator in backward pass
                errD_real.backward()    
                D_x = output.mean().item()

                # 1.2 train with all-fake batch
                # generate batch of latent vectors aka noise
                noise = torch.randn(batch_size, self.latent_vector_size, 1, 1, device=self.device)

                # generater fake image batch with generator
                fake = self.generator(noise)
                label.fill_(fake_label)

                # forward pass fake through discriminator
                output = self.discriminator(fake.detach()).view(-1)
                
                # calc loss all-fake batch
                errD_fake = self.loss(output, label)
                # calc gradients for discriminator in backward pass
                errD_fake.backward()    
                D_G_z1 = output.mean().item()

                # add gradients from all-real and all-fake
                errD = errD_real + errD_fake
                # update discriminator
                self.optim_D.step()

            # 2 update generator
                self.generator.zero_grad()
                label.fill_(real_label)

                # forward pass fake through discriminator, because it got updated
                output = self.discriminator(fake).view(-1)

                # calc loss
                errG = self.loss(output, label)

                # calc gradient for generator using backward pass
                errG.backward()
                D_G_z2 = output.mean().item()
                # update generator
                self.optim_G.step()

            # 3. tracking
                # Output training stats
                if i % 50 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                        % (epoch, self.epochs, i, len(self.dataloader),
                            errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                
                    # write_data = np.moveaxis(fake[0].detach().cpu().numpy(), 0, -1) * 255          
                    # im = Image.fromarray(np.uint8(write_data), mode='RGBA')

                    # img_path = os.getcwd() + '/drive/My Drive/DGM'
                    # im.save(f'{img_path}/imgs/{epoch}_fake.png')
                    
                    
                    # images = [Image.fromarray(np.uint8(x), mode='RGBA') for x in np.moveaxis(fake[:10].detach().cpu().numpy(), 0, -1) * 255]
                    # widths, heights = zip(*(i.size for i in images))

                    # total_width = sum(widths)
                    # max_height = max(heights)

                    # new_im = Image.new('RGB', (total_width, max_height))

                    # x_offset = 0
                    # for im in images:
                    #     new_im.paste(im, (x_offset,0))
                    #     x_offset += im.size[0]
                    
                    # img_path = os.getcwd() + '/drive/My Drive/DGM'
                    # new_im.save(f'{img_path}/imgs/fake_{epoch}.png')

                # # Save Losses for plotting later
                # G_losses.append(errG.item())
                # D_losses.append(errD.item())
                
                # Check how the generator is doing by saving G's output on fixed_noise
                # if (iters % 500 == 0) or ((epoch == self.epochs-1) and (i == len(self.dataloader)-1)):
                #     with torch.no_grad():
                #         fake = self.generator(fixed_noise).detach().cpu()
                    # rgba to rgb
                    # img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                    # rgb
                    # img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                    
                # iters += 1

            # 4. visualize
            if epoch % 100 == 0:
                # Grab a batch of real images from the dataloader
                real_batch = next(iter(self.dataloader))
                # Plot the real images
                img_path_real = os.getcwd() + '/drive/My Drive/DGM/imgs/real/'
                vutils.save_image(real_batch[0][:64,:3,:,:].to(self.device)[:64], f'{img_path_real}{epoch}_real.png', padding=3, normalize=False)
                
                # Generate a batch of fake images
                with torch.no_grad():
                    fake = self.generator(fixed_noise).detach().cpu()
                
                model_type = 'wgan' # 'dcgan'
                idf = f'_{self.args.lr}_{self.args.epochs}_{self.args.batch_size}_{self.args.latent_vector_size}_{model_type}_' 
                # Plot the fake images
                img_path_fake = os.getcwd() + '/drive/My Drive/DGM/imgs/fake/'
                vutils.save_image(fake[:64,:3,:,:].to(self.device)[:64], f'{img_path_fake}{epoch}{idf}fake.png', padding=3, normalize=False)

if __name__ == '__main__':
    
    PARSER = argparse.ArgumentParser()

    PARSER.add_argument(
        '--lr',
        help='learning rate',
        default=0.0002,
        type=float
    )
    PARSER.add_argument(
        '--beta1',
        help='adam optimizer beta1',
        default=0.5,
        type=float
    )
    PARSER.add_argument(
        '--beta2',
        help='adam optimizer beta2',
        default=0.999,
        type=float
    )
    PARSER.add_argument(
        '--epochs',
        help='epochs',
        default=1000,
        type=int
    )
    PARSER.add_argument(
        '--batch-size',
        help='batch size',
        default=128,
        type=int
    )
    PARSER.add_argument(
        '--latent-vector-size',
        help='latent vector size',
        default=100,
        type=int
    )
    PARSER.add_argument(
        '--loss',
        help='loss function [wasserstein or BCEloss]',
        default='BCE',
        type=str
    )
    
    ARGUMENTS, _ = PARSER.parse_known_args()

    ash_ketchum = Training(ARGUMENTS)
    ash_ketchum.train()
