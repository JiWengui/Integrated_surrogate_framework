
# 这是用跟网上一样的32x32的图片尺寸输入的版本，可以。
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
for p in time_point:
    os.makedirs('./result/' + p + 'd', exist_ok=True)

parser = argparse.ArgumentParser()          #命令行选项、参数和子命令解析器
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")  #迭代次数
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")          #batch大小
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")            #学习率
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient") #动量梯度下降第一个参数
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient") #动量梯度下降第二个参数
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation") #CPU个数
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")  #噪声数据生成维度
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")  #输入数据的维度
parser.add_argument("--channels", type=int, default=1, help="number of image channels")      #输入数据的通道数
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")  #保存图像的迭代数
opt = parser.parse_args()
print(opt)
#
cuda = True if torch.cuda.is_available() else False        #判断GPU可用，有GPU用GPU，没有用CPU


def weights_init_normal(m):            #自定义初始化参数
    classname = m.__class__.__name__   #获得类名
    if classname.find("Conv") != -1:   #在类classname中检索到了Conv
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 4
        self.bn = nn.Sequential(nn.BatchNorm2d(1),)  # 只进行批归一化
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

        # img = self.conv_blocks(self.x3)
        return ct5


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.bn = nn.Sequential(nn.BatchNorm2d(1), )  # 只进行批归一化
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
adversarial_loss = torch.nn.BCELoss()         #定义了一个BCE损失函数

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:                                #初始化，将数据放在cuda上
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
        k_data = np.array(pd.read_csv('.../all_data/sensitivity/len_10/k_data/第' + str(m) + '个场.txt', header=None, sep=' '))
        k_data = k_data
        max = np.max(k_data)
        min = np.min(k_data)
        k_normal = (k_data-min)/(max-min)   # 归一化数据，分布在0-1之间
        with open('.../all_data/sensitivity/len_10/ion/'+t_p+'d/no_'+str(m)+'_'+t_p+'天化学结果.pkl', 'rb') as f:
            ion_data = pickle.load(f)
            ion_data = ion_data
            max = np.max(ion_data)
            min = np.min(ion_data)
            ion_normal = (ion_data - min) / (max - min)
        k_input.append(k_normal)
        # k_input.append(k_data)  # 用的是没有进行归一化的渗透系数数据
        x_train.append(ion_normal)
a = 0

# Optimizers                             定义神经网络的优化器  Adam就是一种优化器
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------
x_guding = [30,34,37,42,43,40,36,32,34,40]
y_guding = [36,38,36,36,32,29,32,32,35,33]
well_all = []
# bianhao = [1, 2, 3]
for idx in range(10):
    well = []
    for i in range(64):
        tem = []
        for j in range(64):
            if i in x_guding and j in y_guding:
                if idx == 0:
                    tem.append(0.55)# newh:0.55.  newnewh:0.91
                if idx == 1:
                    tem.append(0.60)
                if idx == 2:
                    tem.append(0.65)
                if idx == 3:
                    tem.append(0.70)
                if idx == 4:
                    tem.append(0.75)
                if idx == 5:
                    tem.append(0.80)
                if idx == 6:
                    tem.append(0.85)
                if idx == 7:
                    tem.append(0.90)
                if idx == 8:
                    tem.append(0.95)
                if idx == 9:
                    tem.append(1.00)

                # tem.append(1)
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
        # if i >= 2000 and i <2200:
        #     wel = well_all[10]
        # if i >= 2200 and i < 2400:
        #     wel = well_all[11]
        # if i >= 2400 and i <2600:
        #     wel = well_all[12]
        # if i >= 2600 and i < 2800:
        #     wel = well_all[13]
        # if i >= 2800:
        #     wel = well_all[14]

        imgs = np.expand_dims(np.expand_dims(imgs, axis=0), axis=1)

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(Tensor(imgs))     #将真实的图片转化为神经网络可以处理的变量
        # label = real_imgs  # 用真实图作为条件,这是用结果图片作为输入，是错的
        label = Variable(Tensor(np.expand_dims(np.expand_dims(k_input[i], axis=0), axis=1)))    # 用真实图作为条件,这是用渗透系数作为约束条件才是对的

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()   #把梯度置零  每次训练都将上一次的梯度置零，避免上一次的干扰
        gen_imgs = generator(wel, label)           #得到一个批次的图片

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(wel, gen_imgs, label), valid)

        g_loss.backward()         #反向传播和模型更新
        optimizer_G.step()


        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(wel, real_imgs, label), valid)  # 判别器判别真实图片是真的的损失
        fake_loss = adversarial_loss(discriminator(wel, gen_imgs.detach(), label), fake)  # 判别器判别假图片是假的的损失
        d_loss = (real_loss + fake_loss) / 2  # 判别器去判别真实图片是真的和生成图片是假的的损失之和，让这个和越大，说明判别器越准确
        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"

            % (epoch, opt.n_epochs, i, len(x_train), d_loss.item(), g_loss.item())
        )
        batches_done = epoch * len(x_train) + i
        # 5天
        if (epoch+1) % 25 == 0 and i == 199:
            torch.save(generator.state_dict(), './result/15d/newh_G_{}model.pt'.format(epoch))  # save generator
            # torch.save(discriminator.state_dict(), "./imgs_all/102/D_{}model.pt".format(epoch))
        # 10天
        if (epoch + 1) % 25 == 0 and i == 399:
            torch.save(generator.state_dict(), './result/102d/newh_G_{}model.pt'.format(epoch))
        # 15天
        if (epoch + 1) % 25 == 0 and i == 599:
            torch.save(generator.state_dict(), './result/152d/newh_G_{}model.pt'.format(epoch))  # 模型保存
        # 30天
        if (epoch + 1) % 25 == 0 and i == 799:
            torch.save(generator.state_dict(), './result/201d/newh_G_{}model.pt'.format(epoch))  # 模型保存
                # torch.save(discriminator.state_dict(), "./imgs_all/102/D_{}model.pt".format(epoch))  # 模型保存
        # 60天
        if (epoch + 1) % 25 == 0 and i == 999:
            torch.save(generator.state_dict(), './result/251d/newh_G_{}model.pt'.format(epoch))  # 模型保存
        aa = 0
        # 102天
        if (epoch + 1) % 25 == 0 and i == 1199:
            torch.save(generator.state_dict(), './result/300d/newh_G_{}model.pt'.format(epoch))  # 模型保存
        # 152天
        if (epoch + 1) % 25 == 0 and i == 1399:
            torch.save(generator.state_dict(), './result/350d/newh_G_{}model.pt'.format(epoch))  # 模型保存
        # 201天
        if (epoch + 1) % 25 == 0 and i == 1599:
            torch.save(generator.state_dict(), './result/450d/newh_G_{}model.pt'.format(epoch))  # 模型保存
        # 251天
        if (epoch + 1) % 25 == 0 and i == 1799:
            torch.save(generator.state_dict(), './result/551d/newh_G_{}model.pt'.format(epoch))  # 模型保存
        # 300天
        if (epoch + 1) % 25 == 0 and i == 1999:
            torch.save(generator.state_dict(), './result/650d/newh_G_{}model.pt'.format(epoch))  # 模型保存
