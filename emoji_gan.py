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

names = ["grinning",
        "smiley",
        "smile",
        "grin",
        "laughing",
        "sweat_smile",
        "rolling_on_the_floor_laughing",
        "joy",
        "slightly_smiling_face",
        "upside_down_face",
        "wink",
        "blush",
        "innocent",
        "smiling_face_with_3_hearts",
        "heart_eyes",
        "star-struck",
        "kissing_heart",
        "kissing",
        "relaxed",
        "kissing_closed_eyes",
        "kissing_smiling_eyes",
        "yum",
        "stuck_out_tongue",
        "stuck_out_tongue_winking_eye",
        "zany_face",
        "stuck_out_tongue_closed_eyes",
        "money_mouth_face",
        "hugging_face",
        "face_with_hand_over_mouth",
        "shushing_face",
        "thinking_face",
        "zipper_mouth_face",
        "face_with_raised_eyebrow",
        "neutral_face",
        "expressionless",
        "no_mouth",
        "smirk",
        "unamused",
        "face_with_rolling_eyes",
        "grimacing",
        "lying_face",
        "relieved",
        "pensive",
        "sleepy",
        "drooling_face",
        "sleeping",
        "mask",
        "face_with_thermometer",
        "face_with_head_bandage",
        "nauseated_face",
        "face_vomiting",
        "sneezing_face",
        "hot_face",
        "cold_face",
        "woozy_face",
        "dizzy_face",
        "exploding_head",
        "face_with_cowboy_hat",
        "partying_face",
        "sunglasses",
        "nerd_face",
        "face_with_monocle",
        "confused",
        "worried",
        "slightly_frowning_face",
        "white_frowning_face",
        "open_mouth",
        "hushed",
        "astonished",
        "flushed",
        "pleading_face",
        "frowning",
        "anguished",
        "fearful",
        "cold_sweat",
        "disappointed_relieved",
        "cry",
        "sob",
        "scream",
        "confounded",
        "persevere",
        "disappointed",
        "sweat",
        "weary",
        "tired_face",
        "yawning_face",
        "triumph",
        "rage",
        "angry",
        "face_with_symbols_on_mouth",
        "smiling_imp",
        "imp",
        "skull",
        "skull_and_crossbones",
        "hankey",
        "clown_face",
        "japanese_ogre",
        "japanese_goblin",
        "ghost",
        "alien",
        "space_invader",
        "robot_face",
        "smiley_cat",
        "smile_cat",
        "joy_cat",
        "heart_eyes_cat",
        "smirk_cat",
        "kissing_cat",
        "scream_cat",
        "crying_cat_face",
        "pouting_cat",
        "see_no_evil",
        "hear_no_evil",
        "speak_no_evil"]
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv = nn.Sequential(Res_Block_Down_2D(4, 64, 3, 1, nn.LeakyReLU(0.2, inplace=True), False),
                                  Res_Block_Down_2D(
                                      64, 64, 3, 1, nn.LeakyReLU(0.2, inplace=True), True),
                                      Res_Block_Down_2D(64, 1, 3, 1, nn.LeakyReLU(0.2, inplace=True), True))

        self.predict = nn.Sequential(nn.Linear(256, 1))

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.predict(x)
        return x


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        """
        self.conv = nn.Sequential(Res_Block_Up_2D(4, 64, 3, 1, nn.LeakyReLU(0.2, inplace=True)),
                                  Res_Block_Up_2D(
                                      64, 64, 3, 1, nn.LeakyReLU(0.2, inplace=True)),
                                  Res_Block_Up_2D(
                                      64, 64, 3, 1, nn.LeakyReLU(0.2, inplace=True)),
                                  Res_Block_Down_2D(
                                      64, 64, 3, 1, nn.LeakyReLU(0.2, inplace=True), False),
                                  Res_Block_Down_2D(
                                      64, 64, 3, 1, nn.LeakyReLU(0.2, inplace=True), False),
                                  Res_Block_Down_2D(64, 4, 3, 1, nn.Sigmoid(), False))
        """
        nc = 4
        ngf = 64
        self.conv = nn.Sequential(

            nn.ConvTranspose2d( nc, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.Conv2d(ngf * 8, ngf * 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.Conv2d( ngf * 2, ngf, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Sigmoid()
 
        )


    def forward(self, x):
        x = self.conv(x)  # * 255
        return x


class Training():
    def __init__(self, lr=1e-4 * 2):
        self.discriminator = Discriminator().to(device)
        self.generator = Generator().to(device)
        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
        self.discriminator.apply(weights_init)
        self.generator.apply(weights_init)

        self.optim_disc = optim.Adam(self.discriminator.parameters(), lr=lr)
        self.optim_gen = optim.Adam(self.generator.parameters(), lr=lr)
        def loss_d_wg(real, fake):
            return  -(torch.mean(real) - torch.mean(fake))
        def loss_g_wg(fake):
            return  -torch.mean(fake)

        # Loss function
        self.loss_D = loss_d_wg
        self.loss_G = loss_g_wg
        self.steps_disc = 1
        self.steps_gen = 1

        self.epochs = 10000
        self.image_shape = (64, 64)
        self.batch_size = 64
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
                for _ in range(1):   
                    noise = torch.randn(
                        self.batch_size, 4, 8, 8, requires_grad=False).to(device)
                    fake_images = self.generator(noise).to(device)
                    real_images = real_images.to(device)

                    write_data_unordered = (np.moveaxis(fake_images[0].detach().cpu().numpy(), 0, -1) * 255)
                    write_data = np.concatenate((np.expand_dims(write_data_unordered[:,:,2], axis=2), np.expand_dims(write_data_unordered[:,:,1], axis=2), 
                    np.expand_dims(write_data_unordered[:,:,0], axis=2), np.expand_dims(write_data_unordered[:,:,3], axis=2)), axis=2)     
                    im = Image.fromarray(np.uint8(write_data), mode='RGBA')
                    im.save('out_images/out' + str((epoch * 7+i) % 50) + '.png')
                    # Generator , needs update for steps
                    self.optim_gen.zero_grad()
                    loss_gen = self.loss_G(self.discriminator(fake_images))

                    loss_gen.backward()
                    self.optim_gen.step()
                for _ in range(1):   
                # Discrimenator , needs update for steps
                    self.optim_disc.zero_grad()
                    loss_disc = self.loss_D(self.discriminator(real_images), self.discriminator(fake_images.detach()))


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
                #if entry['category'] == 'Smileys & Emotion': # or entry['category'] == 'People & Body':
                if entry['short_name'] in names:
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
