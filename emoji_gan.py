from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch
import torchvision
device = 'cpu'


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv = nn.Sequential(Res_Block_Down_2D(3, 16, 3, 1, nn.SELU(), False),
                                  Res_Block_Down_2D(
                                      16, 16, 3, 1, nn.SELU(), False),
                                  Res_Block_Down_2D(16, 16, 3, 1, nn.SELU(), False))

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
        self.conv = nn.Sequential(Res_Block_Down_2D(1, 16, 3, 1, nn.SELU(), False),
                                  Res_Block_Down_2D(
                                      16, 16, 3, 1, nn.SELU(), False),
                                  Res_Block_Down_2D(16, 3, 3, 1, nn.Sigmoid(), False))

    def forward(self, x):
        x = self.conv(x)  # * 255
        return x


class Training():
    def __init__(self, lr=1e-4):
        self.discriminator = Discriminator().to(device)
        self.generator = Generator().to(device)

        self.optim_disc = optim.Adam(self.discriminator.parameters(), lr=lr)
        self.optim_gen = optim.Adam(self.generator.parameters(), lr=lr)

        # Loss function
        self.loss = nn.BCELoss()

        self.steps_disc = 1
        self.steps_gen = 1

        self.epochs = 1
        self.image_shape = (64, 64)
        self.batch_size = 16
        self.dataloader = self.load_dataset()

    def load_dataset(self):
        data_path = 'node_modules/emoji-datasource-apple/img/apple/'
        train_dataset = torchvision.datasets.ImageFolder(
            root=data_path,
            transform=torchvision.transforms.ToTensor()
        )
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
                noise = torch.randn(
                    self.batch_size, 1, self.image_shape[0], self.image_shape[1], requires_grad=False).to(device)
                fake_images = self.generator(noise).to(device)
                real_images = real_images.to(device)
                plt.imshow(fake_images[0].reshape(
                    self.image_shape[0], self.image_shape[1], 3).detach().cpu().numpy())
                plt.savefig('out.png')
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


##############################################################################

if __name__ == "__main__":
    Training().start()
