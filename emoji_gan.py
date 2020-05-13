from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import numpy as np
import cv2
import os
from PIL import Image
import json
device = 'cuda'


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv = nn.Sequential(Res_Block_Down_2D(4, 64, 3, 1, nn.Sigmoid(), False),
                                  Res_Block_Down_2D(
                                      64, 64, 3, 1, nn.Sigmoid(), False),
                                      Res_Block_Down_2D(64, 16, 3, 1, nn.Sigmoid(), False))

        self.predict = nn.Sequential(nn.Linear(16, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.conv(x)
        channel_max = torch.squeeze(
            F.max_pool2d(x, kernel_size=x.size()[2:]))
        x = self.predict(channel_max)
        return x


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv = nn.Sequential(Res_Block_Up_2D(4, 64, 3, 1, nn.Sigmoid()),
                                  Res_Block_Up_2D(
                                      64, 64, 3, 1, nn.Sigmoid()),
                                  Res_Block_Up_2D(
                                      64, 64, 3, 1, nn.Sigmoid()),
                                  Res_Block_Up_2D(
                                      64, 64, 3, 1, nn.Sigmoid()),
                                  Res_Block_Up_2D(
                                      64, 16, 3, 1, nn.Sigmoid()),
                                  Res_Block_Down_2D(16, 4, 3, 1, nn.Sigmoid(), False))

    def forward(self, x):
        x = self.conv(x)  # * 255
        return x


class Training():
    def __init__(self, lr=1e-5):
        self.discriminator = Discriminator().to(device)
        self.generator = Generator().to(device)

        self.optim_disc = optim.Adam(self.discriminator.parameters(), lr=lr)
        self.optim_gen = optim.Adam(self.generator.parameters(), lr=lr)

        # Loss function
        self.loss = nn.BCELoss()

        self.steps_disc = 1
        self.steps_gen = 1

        self.epochs = 10000
        self.image_shape = (64, 64)
        self.batch_size = 128
        self.dataloader = self.load_dataset()

    def load_dataset(self):
        train_dataset = EmojiDataset()
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=True
        )
        return train_loader

    def start(self):
        ones = torch.ones(self.batch_size, requires_grad=False).to(device)
        zeros = torch.zeros(self.batch_size, requires_grad=False).to(device)
        for epoch in range(self.epochs):
            for i, (real_images, _) in enumerate(self.dataloader):
                if real_images.shape[0] < self.batch_size:
                    continue
                for _ in range(2):   
                    noise = torch.randn(
                        self.batch_size, 4, 2, 2, requires_grad=False).to(device)
                    fake_images = self.generator(noise).to(device)
                    real_images = real_images.to(device)

                    write_data = np.moveaxis(fake_images[0].detach().cpu().numpy(), 0, -1) * 255          
                    im = Image.fromarray(np.uint8(write_data), mode='RGBA')
                    im.save('out.png')
                    # Generator , needs update for steps
                    self.optim_gen.zero_grad()
                    loss_gen = self.loss(self.discriminator(fake_images),
                                        ones)

                    loss_gen.backward()
                    self.optim_gen.step()

                # Discrimenator , needs update for steps
                self.optim_disc.zero_grad()

                loss_fake = self.loss(self.discriminator(fake_images.detach()),
                                      zeros)
                loss_real = self.loss(self.discriminator(real_images),
                                      ones)

                loss_disc = (loss_fake + loss_real) / 2

                loss_disc.backward()
                self.optim_disc.step()

                print("Epoch", epoch, "Iteration", i, "Generator loss",
                      loss_gen.item(), "Discriminator loss", loss_disc.item())

class EmojiDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self):
        paths = ['node_modules/emoji-datasource-apple/img/apple/64/', 'node_modules/emoji-datasource-twitter/img/twitter/64/',
        'node_modules/emoji-datasource-facebook/img/facebook/64/', 'node_modules/emoji-datasource-google/img/google/64/']
        arr_file_names = []
        files = []
        with open('node_modules/emoji-datasource-apple/emoji_pretty.json') as json_file:
            data = json.load(json_file)
            for entry in data:
                if entry['category'] == 'Smileys & Emotion':# or entry['category'] == 'People & Body':
                    files.append(entry['image'])
        images = []
        for file in files:
            for path in paths:
                im = np.array(cv2.imread(path + file, cv2.IMREAD_UNCHANGED))
                try:
                    im_rgba = im / 255.0
                    im_rgba = np.moveaxis(im_rgba, -1, 0)
                    if im_rgba.shape[0] == 3:
                        continue
                    images.append(im_rgba)
                except:
                    continue

        self.data = np.array(images)
        print(self.data.shape)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]).float(), torch.zeros(1)

class Res_Block_Down_2D(nn.Module):
    def __init__(self, size_in_channels, size_out_channels, size_filter, size_stride, fn_act, pool_avg):
        super(Res_Block_Down_2D, self).__init__()

        # Params +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self._pool_avg = pool_avg
        self._size_in_channels = size_in_channels
        self._size_out_channels = size_out_channels

        # Nodes ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        self.layer_conv1 = nn.Conv2d(size_in_channels, size_out_channels, size_filter, size_stride, padding=(
            int(size_filter/2), int(size_filter/2)))
        self.layer_norm1 = nn.BatchNorm2d(size_out_channels)

        self.fn_act = fn_act
        self.fn_identity = nn.Identity()

        self.layer_conv2 = nn.Conv2d(size_out_channels, size_out_channels, size_filter, size_stride, padding=(
            int(size_filter/2), int(size_filter/2)))
        self.layer_norm2 = nn.BatchNorm2d(size_out_channels)

        self.channel_conv = nn.Conv2d(
            size_in_channels, size_out_channels, 1, 1)

        if self._pool_avg:
            self.layer_pool = nn.AvgPool2d((2, 2), stride=2)

    def forward(self, x):
        identity = self.fn_identity(x)

        out = self.layer_conv1(x)
        out = self.layer_norm1(out)
        out = self.fn_act(out)
        out = self.layer_conv2(out)
        out = self.layer_norm2(out)

        identity = self.channel_conv(identity)
        out += identity
        out = self.fn_act(out)

        if self._pool_avg:
            out = self.layer_pool(out)

        return out


class Res_Block_Up_2D(nn.Module):
    def __init__(self, size_in_channels, size_out_channels, size_filter, size_stride, fn_act):
        super(Res_Block_Up_2D, self).__init__()

        # Nodes ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        self.layer_conv1 = nn.Conv2d(size_in_channels,size_in_channels, size_filter, size_stride,  padding=(int(size_filter/2),int(size_filter/2)))
        self.layer_norm1 = nn.BatchNorm2d(size_in_channels)

        self.fn_act = fn_act
        self.fn_identity = nn.Identity()

        self.layer_conv2= nn.Conv2d(size_in_channels,size_in_channels, size_filter, size_stride, padding=(int(size_filter/2),int(size_filter/2)))
        self.layer_norm2 = nn.BatchNorm2d(size_in_channels)

        self.layer_up = nn.ConvTranspose2d(size_in_channels, size_out_channels, size_filter + 1, (2,2), padding=(1,1))


    def forward(self, x):
        identity = self.fn_identity(x)

        out = self.layer_conv1(x)
        out = self.layer_norm1(out)
        out = self.fn_act(out)
        out = self.layer_conv2(out)
        out = self.layer_norm2(out)

        out += identity
        out = self.layer_up(out)
        out = self.fn_act(out)
        return out

##############################################################################

if __name__ == "__main__":
    Training().start()
