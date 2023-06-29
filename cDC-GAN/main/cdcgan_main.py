
import argparse
import os
import numpy as np
import pandas as pd
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch

time_point = ['5','10','15','30','60', '102', '152','201','251','300','350','450','551','650']
# for p in time_point:
#     os.makedirs('./result/' + p + 'd', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")  #epochs
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")          #batch size
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")            #learning rate
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation") #CPU number
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")  #input dimension
parser.add_argument("--channels", type=int, default=1, help="number of image channels")      #input channel
opt = parser.parse_args()
print(opt)
#
cuda = True if torch.cuda.is_available() else False

def weights_init_normal(m):            # Initialization parameters
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.init_size = opt.img_size // 4
        self.bn = nn.Sequential(nn.BatchNorm2d(1),)  #
        self.label1 = nn.Sequential(nn.Conv2d(2, 64, 4, 2, 1), nn.BatchNorm2d(64, 0.8), nn.LeakyReLU(0.2,inplace=True))  # 32
        self.label2 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128, 0.8),nn.LeakyReLU(0.2,inplace=True))  # 16
        self.label3 = nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256, 0.8), nn.LeakyReLU(0.2,inplace=True))  # 8
        self.label4 = nn.Sequential(nn.Conv2d(256, 512, 4, 2, 1), nn.BatchNorm2d(512, 0.8), nn.LeakyReLU(0.2,inplace=True))  # 4
        self.label5 = nn.Sequential(nn.Conv2d(512, 1024, 4, 1, 0), nn.LeakyReLU(0.2,inplace=True))  # 1

        self.ct1 = nn.Sequential(nn.ConvTranspose2d(1024, 512, 4, 1, 0), nn.BatchNorm2d(512, 0.8), nn.LeakyReLU(0.2,inplace=True))  # 4
        self.ct2 = nn.Sequential(nn.ConvTranspose2d(1024, 256, 4, 2, 1), nn.BatchNorm2d(256, 0.8), nn.LeakyReLU(0.2,inplace=True))  # 8
        self.ct3 = nn.Sequential(nn.ConvTranspose2d(512, 128, 4, 2, 1), nn.BatchNorm2d(128, 0.8), nn.LeakyReLU(0.2,inplace=True))  # 16
        self.ct4 = nn.Sequential(nn.ConvTranspose2d(256, 64, 4, 2, 1), nn.BatchNorm2d(64, 0.8), nn.LeakyReLU(0.2,inplace=True))  # 32
        self.ct5 = nn.Sequential(nn.ConvTranspose2d(128, 1, 4, 2, 1), nn.Tanh(), )  # 64
    def forward(self, well, label):
        label = torch.concat([well, label], dim=1)
        label1 = self.label1(label)
        label2 = self.label2(label1)
        label3 = self.label3(label2)
        label4 = self.label4(label3)
        label5 = self.label5(label4)
        ct1 = self.ct1(label5)
        input_2 = torch.concat([label4, ct1], dim=1)
        ct2 = self.ct2(input_2)
        input_3 = torch.concat([label3, ct2], dim=1)
        ct3 = self.ct3(input_3)
        input_4 = torch.concat([label2, ct3], dim=1)
        ct4 = self.ct4(input_4)
        input_5 = torch.concat([label1, ct4], dim=1)
        ct5 = self.ct5(input_5)
        return ct5

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.bn = nn.Sequential(nn.BatchNorm2d(1), )
        self.label1 = nn.Sequential(
                                    nn.Conv2d(32, 128, 4, 2, 1), nn.BatchNorm2d(128, 0.8),
                                    nn.LeakyReLU(0.2, inplace=True),nn.Dropout2d(0.25),  # 16
                                    nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256, 0.8),
                                    nn.LeakyReLU(0.2, inplace=True),nn.Dropout2d(0.25),  # 8
                                    nn.Conv2d(256, 512, 4, 2, 1), nn.BatchNorm2d(512, 0.8),
                                    nn.LeakyReLU(0.2, inplace=True),nn.Dropout2d(0.25),  # 4
                                    nn.Conv2d(512, 1024, 4, 1, 0), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25),)  # 1
        self.pd = nn.Sequential(nn.Linear(1024, 1), nn.Sigmoid())
        self.img_label = nn.Sequential(nn.Conv2d(3, 32, 4, 2, 1), nn.BatchNorm2d(32, 0.8), nn.LeakyReLU(0.2, inplace=True),nn.Dropout2d(0.25),)  # 32
    def forward(self, well, img, label):
        img = torch.concat([well, img, label], dim=1)
        img = self.img_label(img)
        out = self.label1(img)
        out = out.view(out.shape[0], -1)
        validity = self.pd(out)
        return validity

# Loss function
adversarial_loss = torch.nn.BCELoss()         # loss function

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Configure data loader
import pickle
k_input = []
x_train = []
time_point = ['15','102', '152','201','251','300','350','450','551','650']
for t_p in time_point:
    for m in range(200):
        k_data = np.array(pd.read_csv('./k_data/第' + str(m) + '个场.txt', header=None, sep=' '))
        k_data = k_data
        max = np.max(k_data)
        min = np.min(k_data)
        k_normal = (k_data-min)/(max-min)   # 归一化数据，分布在0-1之间
        with open('../all_data/main/ion/'+t_p+'d/no_'+str(m)+'_'+t_p+'天化学结果.pkl', 'rb') as f:
            ion_data = pickle.load(f)
            ion_data = ion_data
            max = np.max(ion_data)
            min = np.min(ion_data)
            ion_normal = (ion_data - min) / (max - min)
        k_input.append(k_normal)
        x_train.append(ion_normal)
a = 0

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
x_guding = [30,34,37,42,43,40,36,32,34,40]
y_guding = [36,38,36,36,32,29,32,32,35,33]
well_all = []
for idx in range(10):
    well = []
    for i in range(64):
        tem = []
        for j in range(64):
            if i in x_guding and j in y_guding:
                if idx == 0:
                    tem.append(0.55)  # condition--time data
                if idx == 1:
                    tem.append(0.6)
                if idx == 2:
                    tem.append(0.65)
                if idx == 3:
                    tem.append(0.70)
                if idx == 4:
                    tem.append(0.95)
                if idx == 5:
                    tem.append(0.96)
                if idx == 6:
                    tem.append(0.97)
                if idx == 7:
                    tem.append(0.98)
                if idx == 8:
                    tem.append(0.99)
                if idx == 9:
                    tem.append(1.00)
                continue
            else:
                tem.append(0.0)

        well.append(tem)
    well = Variable(Tensor(np.expand_dims(np.expand_dims(np.array(well), axis=0), axis=1)))
    well_all.append(well)

for epoch in range(opt.n_epochs):
    for i, imgs in enumerate(x_train):
        if i < 200:
            wel = well_all[0]
        if i >= 200 and i < 400:
            wel = well_all[1]
        if i >= 400 and i <600:
            wel = well_all[2]
        if i >= 600 and i < 800:
            wel = well_all[3]
        if i >= 800 and i <1000:
            wel = well_all[4]
        if i >= 1000 and i < 1200:
            wel = well_all[5]
        if i >= 1200 and i <1400:
            wel = well_all[6]
        if i >= 1400 and i < 1600:
            wel = well_all[7]
        if i >= 1600 and i <1800:
            wel = well_all[8]
        if i >= 1800 and i < 2000:
            wel = well_all[9]


        imgs = np.expand_dims(np.expand_dims(imgs, axis=0), axis=1)

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(Tensor(imgs))     #
        label = Variable(Tensor(np.expand_dims(np.expand_dims(k_input[i], axis=0), axis=1)))    # add condition


        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()
        gen_imgs = generator(wel, label)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(wel, gen_imgs, label), valid)

        g_loss.backward()         #back error
        optimizer_G.step()


        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(wel, real_imgs, label), valid)  # real figure error
        fake_loss = adversarial_loss(discriminator(wel, gen_imgs.detach(), label), fake)  # fake figure error
        d_loss = (real_loss + fake_loss) / 2  # 判别器去判别真实图片是真的和生成图片是假的的损失之和，让这个和越大，说明判别器越准确
        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"

            % (epoch, opt.n_epochs, i, len(x_train), d_loss.item(), g_loss.item())
        )
        batches_done = epoch * len(x_train) + i

        ##### save model
        if (epoch+1) % 25 == 0 and i == 199:    # 5 d
            torch.save(generator.state_dict(), './result/15d/newnewh_G_{}model.pt'.format(epoch))  # generator
            # torch.save(discriminator.state_dict(), "./imgs_all/102/D_{}model.pt".format(epoch))  # discriminator
        if (epoch + 1) % 25 == 0 and i == 399:  # 102d
            torch.save(generator.state_dict(), './result/102d/newnewh_G_{}model.pt'.format(epoch))
        if (epoch + 1) % 25 == 0 and i == 599:        # 152d
            torch.save(generator.state_dict(), './result/152d/newnewh_G_{}model.pt'.format(epoch))
        if (epoch + 1) % 25 == 0 and i == 799:        # 201
            torch.save(generator.state_dict(), './result/201d/newnewh_G_{}model.pt'.format(epoch))
        if (epoch + 1) % 25 == 0 and i == 999:        # 251d
            torch.save(generator.state_dict(), './result/251d/newnewh_G_{}model.pt'.format(epoch))
        if (epoch + 1) % 25 == 0 and i == 1199:        # 300d
            torch.save(generator.state_dict(), './result/300d/newnewh_G_{}model.pt'.format(epoch))
        if (epoch + 1) % 25 == 0 and i == 1399:        # 350d
            torch.save(generator.state_dict(), './result/350d/newnewh_G_{}model.pt'.format(epoch))
        if (epoch + 1) % 25 == 0 and i == 1599:        # 450d
            torch.save(generator.state_dict(), './result/450d/newnewh_G_{}model.pt'.format(epoch))
        if (epoch + 1) % 25 == 0 and i == 1799:        # 551 d
            torch.save(generator.state_dict(), './result/551d/newnewh_G_{}model.pt'.format(epoch))
        if (epoch + 1) % 25 == 0 and i == 1999: # 650 d
            torch.save(generator.state_dict(), './result/650d/newnewh_G_{}model.pt'.format(epoch))
