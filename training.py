from dataloader import EmojiDataset
from networks import Generator, Discriminator
import torch
from torch import optim, nn
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class Training():

    def __init__(self):
        
        self.latent_vector_size = 100
        self.image_channels = 4

        self.generator = Generator(self.image_channels, self.latent_vector_size, 64)
        self.discriminator = Discriminator(self.image_channels, 64)

        self.generator.apply(self.weights_init)
        self.discriminator.apply(self.weights_init)

        self.optim_G = optim.Adam(self.generator.parameters())
        self.optim_D = optim.Adam(self.discriminator.parameters())
        
        self.loss = nn.BCELoss()

        self.epochs = 100
        self.batch_size = 128

        self.img_shape = (64, 64)

        self.dataloader = torch.utils.data.DataLoader(
            EmojiDataset(),
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=True
        )
        

        self.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

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
                
                    write_data = np.moveaxis(fake[0].detach().cpu().numpy(), 0, -1) * 255          
                    im = Image.fromarray(np.uint8(write_data), mode='RGBA')
                    im.save(f'{epoch}_fake.png')
                    
                    # images = [Image.fromarray(np.uint8(x), mode='RGBA') for x in np.moveaxis(fake[:10].detach().cpu().numpy(), 0, -1) * 255]
                    # widths, heights = zip(*(i.size for i in images))

                    # total_width = sum(widths)
                    # max_height = max(heights)

                    # new_im = Image.new('RGB', (total_width, max_height))

                    # x_offset = 0
                    # for im in images:
                    #     new_im.paste(im, (x_offset,0))
                    #     x_offset += im.size[0]

                    # new_im.save('test.jpg')

                # # Save Losses for plotting later
                # G_losses.append(errG.item())
                # D_losses.append(errD.item())
                
                # # Check how the generator is doing by saving G's output on fixed_noise
                # if (iters % 500 == 0) or ((epoch == self.epochs-1) and (i == len(self.dataloader)-1)):
                #     with torch.no_grad():
                #         fake = self.generator(fixed_noise).detach().cpu()
                #         print(fake.shape)
                #     # rgba to rgb
                #     img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                #     # rgb
                #     # img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                    
                # iters += 1

            # # 4. visualize

            #     # Grab a batch of real images from the dataloader
            #     real_batch = next(iter(self.dataloader))

            #     # Plot the real images
            #     plt.figure(figsize=(15,15))
            #     plt.subplot(1,2,1)
            #     plt.axis("off")
            #     plt.title("Real Images")
            #     # plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))
            #     print(real_batch[0].to(self.device)[:64][:,:2,:,:].size)
            #     plt.imsave(f'{epoch}_real.png', np.transpose(vutils.make_grid(real_batch[0].to(self.device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

            #     # Plot the fake images from the last epoch
            #     plt.subplot(1,2,2)
            #     plt.axis("off")
            #     plt.title("Fake Images")
            #     # plt.imshow(np.transpose(img_list[-1],(1,2,0)))
            #     plt.imsave(f'{epoch}_fake.png', np.transpose(img_list[-1],(1,2,0)))
            #     # plt.show()

if __name__ == '__main__':
    ash_ketchum = Training()
    ash_ketchum.train()