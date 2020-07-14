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

"""
        ,#+++++++++++++++++++++++++++++++++++++++++++++++
        "wave",
        "raised_back_of_hand",
        "raised_hand_with_fingers_splayed",
        "hand",
        "spock-hand",
        "ok_hand",
        "pinching_hand",
        "v",
        "crossed_fingers",
        "i_love_you_hand_sign",
        "the_horns",
        "call_me_hand",
        "point_left",
        "point_right",
        "point_up_2",
        "middle_finger",
        "point_down",
        "point_up",
        "+1",
        "-1",
        "fist",
        "facepunch",
        "left-facing_fist",
        "right-facing_fist",
        "clap",
        "raised_hands",
        "open_hands",
        "palms_up_together",
        "handshake",
        "pray",
        "writing_hand",
        "nail_care",
        "selfie",
        "muscle",
        "mechanical_arm",
        "mechanical_leg",
        "leg",
        "foot",
        "ear",
        "ear_with_hearing_aid",
        "nose",
        "brain",
        "tooth",
        "bone",
        "eyes",
        "eye",
        "tongue",
        "lips",
        "baby",
        "child",
        "boy",
        "girl",
        "adult",
        "person_with_blond_hair",
        "man",
        "bearded_person",
        "red_haired_man",
        "curly_haired_man",
        "white_haired_man",
        "bald_man",
        "woman",
        "red_haired_woman",
        "red_haired_person",
        "curly_haired_woman",
        "curly_haired_person",
        "white_haired_woman",
        "white_haired_person",
        "bald_woman",
        "bald_person",
        "blond-haired-woman",
        "blond-haired-man",
        "older_adult",
        "older_man",
        "older_woman",
        "person_frowning",
        "man-frowning",
        "woman-frowning",
        "person_with_pouting_face",
        "man-pouting",
        "woman-pouting",
        "no_good",
        "man-gesturing-no",
        "woman-gesturing-no",
        "ok_woman",
        "man-gesturing-ok",
        "woman-gesturing-ok",
        "information_desk_person",
        "man-tipping-hand",
        "woman-tipping-hand",
        "raising_hand",
        "man-raising-hand",
        "woman-raising-hand",
        "deaf_person",
        "deaf_man",
        "deaf_woman",
        "bow",
        "man-bowing",
        "woman-bowing",
        "face_palm",
        "man-facepalming",
        "woman-facepalming",
        "shrug",
        "man-shrugging",
        "woman-shrugging",
        "health_worker",
        "male-doctor",
        "female-doctor",
        "student",
        "male-student",
        "female-student",
        "teacher",
        "male-teacher",
        "female-teacher",
        "judge",
        "male-judge",
        "female-judge",
        "farmer",
        "male-farmer",
        "female-farmer",
        "cook",
        "male-cook",
        "female-cook",
        "mechanic",
        "male-mechanic",
        "female-mechanic",
        "factory_worker",
        "male-factory-worker",
        "female-factory-worker",
        "office_worker",
        "male-office-worker",
        "female-office-worker",
        "scientist",
        "male-scientist",
        "female-scientist",
        "technologist",
        "male-technologist",
        "female-technologist",
        "singer",
        "male-singer",
        "female-singer",
        "artist",
        "male-artist",
        "female-artist",
        "pilot",
        "male-pilot",
        "female-pilot",
        "astronaut",
        "male-astronaut",
        "female-astronaut",
        "firefighter",
        "male-firefighter",
        "female-firefighter",
        "cop",
        "male-police-officer",
        "female-police-officer",
        "sleuth_or_spy",
        "male-detective",
        "female-detective",
        "guardsman",
        "male-guard",
        "female-guard",
        "construction_worker",
        "male-construction-worker",
        "female-construction-worker",
        "prince",
        "princess",
        "man_with_turban",
        "man-wearing-turban",
        "woman-wearing-turban",
        "man_with_gua_pi_mao",
        "person_with_headscarf",
        "man_in_tuxedo",
        "bride_with_veil",
        "pregnant_woman",
        "breast-feeding",
        "angel",
        "santa",
        "mrs_claus",
        "superhero",
        "male_superhero",
        "female_superhero",
        "supervillain",
        "male_supervillain",
        "female_supervillain",
        "mage",
        "male_mage",
        "female_mage",
        "fairy",
        "male_fairy",
        "female_fairy",
        "vampire",
        "male_vampire",
        "female_vampire",
        "merperson",
        "merman",
        "mermaid",
        "elf",
        "male_elf",
        "female_elf",
        "genie",
        "male_genie",
        "female_genie",
        "zombie",
        "male_zombie",
        "female_zombie",
        "massage",
        "man-getting-massage",
        "woman-getting-massage",
        "haircut",
        "man-getting-haircut",
        "woman-getting-haircut",
        "walking",
        "man-walking",
        "woman-walking",
        "standing_person",
        "man_standing",
        "woman_standing",
        "kneeling_person",
        "man_kneeling",
        "woman_kneeling",
        "person_with_probing_cane",
        "man_with_probing_cane",
        "woman_with_probing_cane",
        "person_in_motorized_wheelchair",
        "man_in_motorized_wheelchair",
        "woman_in_motorized_wheelchair",
        "person_in_manual_wheelchair",
        "man_in_manual_wheelchair",
        "woman_in_manual_wheelchair",
        "runner",
        "man-running",
        "woman-running",
        "dancer",
        "man_dancing",
        "man_in_business_suit_levitating",
        "dancers",
        "man-with-bunny-ears-partying",
        "woman-with-bunny-ears-partying",
        "person_in_steamy_room",
        "man_in_steamy_room",
        "woman_in_steamy_room",
        "person_climbing",
        "man_climbing",
        "woman_climbing",
        "fencer",
        "horse_racing",
        "skier",
        "snowboarder",
        "golfer",
        "man-golfing",
        "woman-golfing",
        "surfer",
        "man-surfing",
        "woman-surfing",
        "rowboat",
        "man-rowing-boat",
        "woman-rowing-boat",
        "swimmer",
        "man-swimming",
        "woman-swimming",
        "person_with_ball",
        "man-bouncing-ball",
        "woman-bouncing-ball",
        "weight_lifter",
        "man-lifting-weights",
        "woman-lifting-weights",
        "bicyclist",
        "man-biking",
        "woman-biking",
        "mountain_bicyclist",
        "man-mountain-biking",
        "woman-mountain-biking",
        "person_doing_cartwheel",
        "man-cartwheeling",
        "woman-cartwheeling",
        "wrestlers",
        "man-wrestling",
        "woman-wrestling",
        "water_polo",
        "man-playing-water-polo",
        "woman-playing-water-polo",
        "handball",
        "man-playing-handball",
        "woman-playing-handball",
        "juggling",
        "man-juggling",
        "woman-juggling",
        "person_in_lotus_position",
        "man_in_lotus_position",
        "woman_in_lotus_position",
        "bath",
        "sleeping_accommodation",
        "people_holding_hands",
        "two_women_holding_hands",
        "couple",
        "two_men_holding_hands",
        "couplekiss",
        "woman-kiss-man",
        "man-kiss-man",
        "woman-kiss-woman",
        "couple_with_heart",
        "woman-heart-man",
        "man-heart-man",
        "woman-heart-woman",
        "family",
        "man-woman-boy",
        "man-woman-girl",
        "man-woman-girl-boy",
        "man-woman-boy-boy",
        "man-woman-girl-girl",
        "man-man-boy",
        "man-man-girl",
        "man-man-girl-boy",
        "man-man-boy-boy",
        "man-man-girl-girl",
        "woman-woman-boy",
        "woman-woman-girl",
        "woman-woman-girl-boy",
        "woman-woman-boy-boy",
        "woman-woman-girl-girl",
        "man-boy",
        "man-boy-boy",
        "man-girl",
        "man-girl-boy",
        "man-girl-girl",
        "woman-boy",
        "woman-boy-boy",
        "woman-girl",
        "woman-girl-boy",
        "woman-girl-girl",
        "speaking_head_in_silhouette",
        "bust_in_silhouette",
        "busts_in_silhouette",
        "footprints", 
         "monkey_face",
        "monkey",
        "gorilla",
        "orangutan",
        "dog",
        "dog2",
        "guide_dog",
        "service_dog",
        "poodle",
        "wolf",
        "fox_face",
        "raccoon",
        "cat",
        "cat2",
        "lion_face",
        "tiger",
        "tiger2",
        "leopard",
        "horse",
        "racehorse",
        "unicorn_face",
        "zebra_face",
        "deer",
        "cow",
        "ox",
        "water_buffalo",
        "cow2",
        "pig",
        "pig2",
        "boar",
        "pig_nose",
        "ram",
        "sheep",
        "goat",
        "dromedary_camel",
        "camel",
        "llama",
        "giraffe_face",
        "elephant",
        "rhinoceros",
        "hippopotamus",
        "mouse",
        "mouse2",
        "rat",
        "hamster",
        "rabbit",
        "rabbit2",
        "chipmunk",
        "hedgehog",
        "bat",
        "bear",
        "koala",
        "panda_face",
        "sloth",
        "otter",
        "skunk",
        "kangaroo",
        "badger",
        "feet",
        "turkey",
        "chicken",
        "rooster",
        "hatching_chick",
        "baby_chick",
        "hatched_chick",
        "bird",
        "penguin",
        "dove_of_peace",
        "eagle",
        "duck",
        "swan",
        "owl",
        "flamingo",
        "peacock",
        "parrot",
        "frog",
        "crocodile",
        "turtle",
        "lizard",
        "snake",
        "dragon_face",
        "dragon",
        "sauropod",
        "t-rex",
        "whale",
        "whale2",
        "dolphin",
        "fish",
        "tropical_fish",
        "blowfish",
        "shark",
        "octopus",
        "shell",
        "snail",
        "butterfly",
        "bug",
        "ant",
        "bee",
        "beetle",
        "cricket",
        "spider",
        "spider_web",
        "scorpion",
        "mosquito",
        "microbe",
        "bouquet",
        "cherry_blossom",
        "white_flower",
        "rosette",
        "rose",
        "wilted_flower",
        "hibiscus",
        "sunflower",
        "blossom",
        "tulip",
        "seedling",
        "evergreen_tree",
        "deciduous_tree",
        "palm_tree",
        "cactus",
        "ear_of_rice",
        "herb",
        "shamrock",
        "four_leaf_clover",
        "maple_leaf",
        "fallen_leaf",
        "leaves"]
        """
class Discriminator(nn.Module):
    def __init__(self, nc=4, blocks_down = 2):
        super(Discriminator, self).__init__()
        """
        self.conv = nn.Sequential(Res_Block_Down_2D(nc,32, 3, 1, nn.LeakyReLU(0.2, inplace=True), False),
                                        *[Res_Block_Down_2D(
                                      32, 32, 3, 1, nn.LeakyReLU(0.2, inplace=True), True) for i in range(blocks_down)],
                                      Res_Block_Down_2D(32, 1, 3, 1, nn.LeakyReLU(0.2, inplace=True), False))
        """
        ngf = 64
        self.conv = nn.Sequential(

            
            nn.Conv2d( nc, ngf, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.PReLU(),

            nn.AvgPool2d((3, 3), stride=3),
            nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.PReLU(),

            nn.AvgPool2d((3, 3), stride=3),
            nn.Conv2d(ngf, 1, 3, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.PReLU() 
        )

        self.predict = nn.Sequential(nn.Linear(196, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.predict(x)
        return x

class GeneratorDown(nn.Module):
    def __init__(self, nc=4, add_channel=1, convs_up=0, convs_down=0):
        super(GeneratorDown, self).__init__()
        activation = nn.SELU
        nc = nc
        ngf = 128
        self.conv1 = nn.Sequential(
            nn.Conv2d( nc, ngf, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf),
            activation(),
            nn.AvgPool2d((3, 3), stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf),
            activation(),
            nn.AvgPool2d((3, 3), stride=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf),
            activation(),
            nn.AvgPool2d((3, 3), stride=2),
        )

    def forward(self, x):
        if x.shape[1] == 3:
            skip_x = torch.cat((x, torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3]).to(device)), dim=1)
        else:
            skip_x = x[:,:3,:,:]

        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        return conv1, conv2, conv3, skip_x

class GeneratorUp(nn.Module):
    def __init__(self, nc=4, add_channel=1, convs_up=0, convs_down=0):
        super(GeneratorUp, self).__init__()
        activation = nn.SELU
        nc = nc
        ngf = 128
        self.up_conv1 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf, 5, 1, 2, bias=False),
            nn.BatchNorm2d(ngf),
            activation(),
        )
        self.up_conv2 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf),
            activation(),
        )
        self.up_conv3 = nn.Sequential(
            nn.Conv2d( ngf, nc + add_channel, 3, 1, 1, bias=True),
        )


    def forward(self, conv3, conv2_other, conv1_other, skip_x_other):
        conv3 = F.interpolate(conv3, (conv2_other.shape[2], conv2_other.shape[3]), mode='nearest')
        up_conv1 = self.up_conv1(torch.cat((conv3, conv2_other), dim=1))

        up_conv1 = F.interpolate(up_conv1, (conv1_other.shape[2], conv1_other.shape[3]), mode='nearest')
        up_conv2 = self.up_conv2(torch.cat((up_conv1, conv1_other), dim=1))

        up_conv2 = F.interpolate(up_conv2, (skip_x_other.shape[2], skip_x_other.shape[3]), mode='nearest')
        y = torch.sigmoid(self.up_conv3(up_conv2))# + skip_x_other)
        return y

class DoubleGenerator(nn.Module):
    def __init__(self):
        super(DoubleGenerator, self).__init__()
        self.generator_up1 = GeneratorUp(nc=3)
        self.generator_up2 = GeneratorUp(nc=4, add_channel=-1)

        self.generator_down1 = GeneratorDown(nc=3)
        self.generator_down2 = GeneratorDown(nc=4, add_channel=-1)

    def forward(self, x1, x2):
        conv1_x1, conv2_x1, conv3_x1, skip_x_x1 = self.generator_down1(x1)
        conv1_x2, conv2_x2, conv3_x2, skip_x_x2 = self.generator_down2(x2)

        y2 = self.generator_up2(conv3_x1, conv2_x2, conv1_x2, skip_x_x2)
        y1 = self.generator_up1(conv3_x2, conv2_x1, conv1_x1, skip_x_x1)

        return y1, y2

class Training():
    def __init__(self, lr=1e-4 * 1):
        self.discriminator1 = Discriminator(nc=4, blocks_down=4).to(device)
        self.discriminator2 = Discriminator(nc=3, blocks_down=4).to(device)
        self.generator = DoubleGenerator().to(device)


        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
        #self.discriminator1.apply(weights_init)
        #self.generator1.apply(weights_init)
        #self.discriminator2.apply(weights_init)
        #self.generator2.apply(weights_init)

        self.optim_disc = optim.Adam(list(self.discriminator1.parameters()) + list(self.discriminator2.parameters()), lr=lr)
        self.optim_gen = optim.Adam(list(self.generator.parameters()), lr=lr)

        def loss_d_wg(real1,real2, fake1, fake2):
            return  (-(torch.mean(real1)+torch.mean(real2)) + (torch.mean(fake1)+torch.mean(fake2)))
        def loss_g_wg(fake1, fake2):
            return  -torch.mean(fake1) - torch.mean(fake2)

        # Loss function
        self.loss_D = loss_d_wg
        self.loss_G = loss_g_wg
        self.loss_R = nn.MSELoss(reduction='mean')
        self.steps_disc = 1
        self.steps_gen = 1

        self.epochs = 10000
        self.image_shape = (64, 64)
        self.batch_size = 6
        self.dataloader = self.load_dataset()

    def load_dataset(self):
        train_dataset = EmojiDataset()
        target_dataset = ImageDataset()
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        target_loader = torch.utils.data.DataLoader(
            target_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        return train_loader, target_loader

    def start(self, load=False):
        if load:
            self.discriminator1.load_state_dict(torch.load("model/" + "disc1.pt"))
            self.discriminator2.load_state_dict(torch.load("model/" + "disc2.pt"))
            self.generator.load_state_dict(torch.load("model/" + "gen.pt"))
            self.optim_disc.load_state_dict(torch.load("optimizer/" + "disc.pt"))
            self.optim_gen.load_state_dict(torch.load("optimizer/"+ "gen.pt"))
        ones = torch.ones(self.batch_size, requires_grad=False).to(device)
        zeros = torch.zeros(self.batch_size, requires_grad=False).to(device)

        for epoch in range(self.epochs):

            iter_emojis = iter(self.dataloader[0]) 
            iter_images = iter(self.dataloader[1])
            num_batches = min(len(self.dataloader[1]), len(self.dataloader[0]))
            for i in range(num_batches):
                emojis = iter_emojis.next()[0].to(device)
                images = iter_images.next()[0].to(device)
                
                if emojis.shape[0] < self.batch_size  or images.shape[0] < self.batch_size:
                    continue

                fake_emojis, fake_images = self.generator(images, emojis)


                for _ in range(1):   
                # Discrimenator , needs update for steps
                    self.optim_disc.zero_grad()
                    loss_disc = self.loss_D(self.discriminator1(emojis), self.discriminator2(images), self.discriminator1(fake_emojis), self.discriminator2(fake_images))

                    loss_disc.backward()
                    self.optim_disc.step()


                # HIGHLY INEFFICIENT. retain graph would be better

                fake_emojis, fake_images = self.generator(images, emojis)
                reconst_emojis, reconst_images = self.generator(fake_images, fake_emojis)

                for _ in range(1):   
                    # Generator , needs update for steps
                    self.optim_gen.zero_grad()
                    loss_gen = self.loss_G(self.discriminator1(fake_emojis), self.discriminator2(fake_images)) + \
                            self.loss_R(images, reconst_images) + self.loss_R(emojis, reconst_emojis)
                    

                    loss_gen.backward()
                    self.optim_gen.step()


                # WRITE DATA
                if i % 15 == 14:
                    write_data_unordered = (np.moveaxis(emojis[0].detach().cpu().numpy(), 0, -1) * 255)
                    write_data = np.concatenate((np.expand_dims(write_data_unordered[:,:,2], axis=2), np.expand_dims(write_data_unordered[:,:,1], axis=2), 
                    np.expand_dims(write_data_unordered[:,:,0], axis=2), np.expand_dims(write_data_unordered[:,:,3], axis=2)), axis=2)     
                    im = Image.fromarray(np.uint8(write_data), mode='RGBA')
                    im.save('out_images/' + 'org_emoji.png')

                    write_data_unordered = (np.moveaxis(fake_emojis[0].detach().cpu().numpy(), 0, -1) * 255)
                    write_data = np.concatenate((np.expand_dims(write_data_unordered[:,:,2], axis=2), np.expand_dims(write_data_unordered[:,:,1], axis=2), 
                    np.expand_dims(write_data_unordered[:,:,0], axis=2), np.expand_dims(write_data_unordered[:,:,3], axis=2)), axis=2)     
                    im = Image.fromarray(np.uint8(write_data), mode='RGBA')
                    im.save('out_images/' + 'fake_emoji.png')

                    write_data_unordered = (np.moveaxis(images[0].detach().cpu().numpy(), 0, -1) * 255)
                    write_data = np.concatenate((np.expand_dims(write_data_unordered[:,:,2], axis=2), np.expand_dims(write_data_unordered[:,:,1], axis=2), 
                    np.expand_dims(write_data_unordered[:,:,0], axis=2)), axis=2)     
                    im = Image.fromarray(np.uint8(write_data), mode='RGB')
                    im.save('out_images/' + 'org_image.png')


                    # Get help and use alpha channel from orignal
                    out_fake_image = (np.moveaxis(fake_images[0].detach().cpu().numpy(), 0, -1) * 255)
                    org_emoji = (np.moveaxis(emojis[0].detach().cpu().numpy(), 0, -1) * 255)
                    write_data_unordered = np.concatenate((out_fake_image[:, :, :3] , np.expand_dims(org_emoji[:, :, 3], axis=2)), axis = 2)

                    write_data = np.concatenate((np.expand_dims(write_data_unordered[:,:,2], axis=2), np.expand_dims(write_data_unordered[:,:,1], axis=2), 
                    np.expand_dims(write_data_unordered[:,:,0], axis=2), np.expand_dims(write_data_unordered[:,:,3], axis=2)), axis=2)     
                    im = Image.fromarray(np.uint8(write_data), mode='RGBA')
                    im.save('out_images/' + 'fake_image.png')

                print("Epoch", epoch, "Iteration", i, "Generator loss",
                      loss_gen.item(), "Discriminator loss", loss_disc.item())
                torch.save(self.discriminator1.state_dict(), "model/"+ "disc1.pt")
                torch.save(self.discriminator2.state_dict(), "model/"+ "disc2.pt")
                torch.save(self.generator.state_dict(), "model/"+ "gen.pt")
                torch.save(self.optim_disc.state_dict(), "optimizer/"+ "disc.pt")
                torch.save(self.optim_gen.state_dict(), "optimizer/"+ "gen.pt")
        
class EmojiDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, up=4):
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
                    im_rgba = cv2.resize(im_rgba,(int(64 * up),int(64*up)))
                    im_rgba = np.moveaxis(im_rgba, -1, 0)
                    if im_rgba.shape[0] == 3:
                        continue
                    images.append(im_rgba)
                except Exception as e:
                    print(e)
                    continue

        self.data = np.array(images)
        self.data_up = np.empty((self.data.shape[0], self.data.shape[1], self.data.shape[2], self.data.shape[3]))

        for i in range(self.data.shape[0]):
            self.data_up[i] = self.data[i]
        self.data = self.data_up
        print(self.data.shape)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]).float(), torch.zeros(1)

class ImageDataset(Dataset):
    def __init__(self, down=4):
        paths = ['01000_hq']
        arr_file_names = []

        images = np.empty((1000, 3, int(1024/down), int(1024/down)))

        for path in paths:
            files = os.listdir(os.getcwd() +'/'+ path)
            for i, file in enumerate(files):
                im = np.array(cv2.imread(path +'/' + file, cv2.IMREAD_UNCHANGED))

                try:
                    im_rgb = im / 255.0
                    im_rgb = np.moveaxis(im_rgb, -1, 0)
                    im_rgb = im_rgb[::,::int(down),::int(down)]
                    images[i] = im_rgb
                except:
                    continue

        self.data = np.array(images)
        print(self.data.shape)
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]).float(), torch.zeros(1)

class ImageDatasetGogh(Dataset):
    def __init__(self, down=4):
        paths = ['gogh/vgdb_2016/train/vg']#, 'gogh/vgdb_2016/test/vg', 'gogh/vgdb_2016/test/nvg'] #, 'gogh/vgdb_2016/train/nvg']
        arr_file_names = []

        images = np.empty((99, 3, int(1024/down), int(1024/down)))

        for path in paths:
            files = os.listdir(os.getcwd() +'/'+ path)
            for i, file in enumerate(files):
                im = np.array(cv2.imread(path +'/' + file, cv2.IMREAD_UNCHANGED))
                im = cv2.resize(im,(int(1024/down),int(1024/down)))

                try:
                    im_rgb = im / 255.0
                    im_rgb = np.moveaxis(im_rgb, -1, 0)
                    images[i] = im_rgb
                except Exception as e:
                    print(e)
                    continue

        self.data = np.array(images)
        for i in range(self.data.shape[0]):
            if np.unique(self.data[i]).shape[0] == 1:
                print(i)
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

class Inference():
    def __init__(self, path=""):
        self.discriminator1 = Discriminator(nc=4, blocks_down=4).to(device)
        self.discriminator2 = Discriminator(nc=3, blocks_down=4).to(device)
        self.generator = DoubleGenerator().to(device)

        # path must contain gen1.pt and gen2.pt
        self.generator.load_state_dict(torch.load(path + "/gen.pt"))
        self.generator.eval()

    
    def apply(self, emoji_path, image_path):
        # Load emoji
        try:
            im = np.array(cv2.imread(emoji_path, cv2.IMREAD_UNCHANGED))
            im_rgba = im / 255.0
            emoji = np.moveaxis(im_rgba, -1, 0).repeat(4, 1).repeat(4, 2)
        except Exception as e:
            print("Emoji not loaded.")
            print(e)

        # Load image
        try:
            down=4
            im = np.array(cv2.imread(image_path, cv2.IMREAD_UNCHANGED))
            im = cv2.resize(im,(int(1024/down),int(1024/down)))
            im_rgb = im / 255.0
            im_rgb = np.moveaxis(im_rgb, -1, 0)
            image = im_rgb
        except Exception as e:
            print("Image not loaded.")
            print(e)

        # To tensor
        emoji = torch.unsqueeze(torch.from_numpy(emoji).to(device),dim=0).float()
        image = torch.unsqueeze(torch.from_numpy(image).to(device),dim=0).float()

        # Rescale image
        image = nn.functional.interpolate(image, size=(256,256), mode='nearest')
        print(image.shape, emoji.shape)
        fake_emoji, fake_image = self.generator(image, emoji)


        write_data_unordered = (np.moveaxis(emoji[0].detach().cpu().numpy(), 0, -1) * 255)
        write_data = np.concatenate((np.expand_dims(write_data_unordered[:,:,2], axis=2), np.expand_dims(write_data_unordered[:,:,1], axis=2), 
        np.expand_dims(write_data_unordered[:,:,0], axis=2), np.expand_dims(write_data_unordered[:,:,3], axis=2)), axis=2)     
        im = Image.fromarray(np.uint8(write_data), mode='RGBA')
        im.save('out_images/' + 'org_emoji.png')

        write_data_unordered = (np.moveaxis(fake_emoji[0].detach().cpu().numpy(), 0, -1) * 255)
        write_data = np.concatenate((np.expand_dims(write_data_unordered[:,:,2], axis=2), np.expand_dims(write_data_unordered[:,:,1], axis=2), 
        np.expand_dims(write_data_unordered[:,:,0], axis=2), np.expand_dims(write_data_unordered[:,:,3], axis=2)), axis=2)     
        im = Image.fromarray(np.uint8(write_data), mode='RGBA')
        im_emo_out = im
        im.save('out_images/' + 'fake_emoji.png')

        write_data_unordered = (np.moveaxis(nn.functional.interpolate(fake_emoji.detach(), (64,64), mode='bilinear')[0].cpu().numpy(), 0, -1) * 255)
        write_data = np.concatenate((np.expand_dims(write_data_unordered[:,:,2], axis=2), np.expand_dims(write_data_unordered[:,:,1], axis=2), 
        np.expand_dims(write_data_unordered[:,:,0], axis=2), np.expand_dims(write_data_unordered[:,:,3], axis=2)), axis=2)     
        im = Image.fromarray(np.uint8(write_data), mode='RGBA')
        im.save('out_images/' + 'fake_emoji_small.png')

        write_data_unordered = (np.moveaxis(image[0].detach().cpu().numpy(), 0, -1) * 255)
        write_data = np.concatenate((np.expand_dims(write_data_unordered[:,:,2], axis=2), np.expand_dims(write_data_unordered[:,:,1], axis=2), 
        np.expand_dims(write_data_unordered[:,:,0], axis=2)), axis=2)     
        im = Image.fromarray(np.uint8(write_data), mode='RGB')
        im.save('out_images/' + 'org_image.png')

        write_data_unordered = (np.moveaxis(fake_image[0].detach().cpu().numpy(), 0, -1) * 255)
        write_data = np.concatenate((np.expand_dims(write_data_unordered[:,:,2], axis=2), np.expand_dims(write_data_unordered[:,:,1], axis=2), 
        np.expand_dims(write_data_unordered[:,:,0], axis=2)), axis=2)     
        im = Image.fromarray(np.uint8(write_data), mode='RGB')
        im_im_out = im
        im.save('out_images/' + 'fake_image.png')

        return im_emo_out, im_im_out

##############################################################################

if __name__ == "__main__":
    #Training().start(load=False) #"node_modules/emoji-datasource-apple/img/apple/64/1f924.png" "01000_hq/01000.png"
    Inference("model").apply("1f602.png", "nicolas.png")