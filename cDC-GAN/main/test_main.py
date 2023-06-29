
import argparse
import os
import numpy as np
import pandas as pd
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn as nn
import pickle
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
opt = parser.parse_args()
print(opt)
cuda = True if torch.cuda.is_available() else False

def weights_init_normal(m):
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
        self.bn = nn.Sequential(nn.BatchNorm2d(1),)
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

# Loss function
adversarial_loss = torch.nn.BCELoss()
# Initialize generator and discriminator
generator = Generator()
if cuda:
    generator.cuda()
    adversarial_loss.cuda()
# Initialize weights
generator.apply(weights_init_normal)
# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

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
                    tem.append(0.55)
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

# Validation Model
star_num = 200
idx = 0
# time_point = ['15','102', '152','201','251','300','350','450','551','650']
time_point = ['551']
for t_p in time_point:
    x_test = []
    k_test = []
    for m in range(60):
        k_data = np.array(pd.read_csv('./k_data/第' + str(m+star_num) + '个场.txt', header=None, sep=' '))
        max = np.max(k_data)
        min = np.min(k_data)
        k_normal = (k_data-min)/(max-min)
        with open('../all_data/main/ion/'+t_p+'d/no_'+str(m + star_num)+'_'+t_p+'天化学结果.pkl', 'rb') as f:
            ion_data = pickle.load(f)
            ion_data = ion_data
            max = np.max(ion_data)
            min = np.min(ion_data)
            ion_normal = (ion_data - min) / (max - min)
        k_test.append(k_normal)
        x_test.append(ion_normal)

    modelGenerator = Generator()
    modelGenerator.cuda()
    # 加载 train.py 里训练好的模型
    # modelGenerator.load_state_dict(torch.load('./result/'+t_p+'d/newh_G_49model.pt'))   # 152d
    # modelGenerator.load_state_dict(torch.load('./result/' + t_p + 'd/newh_G_74model.pt'))  # 300d
    # modelGenerator.load_state_dict(torch.load('./result/' + t_p + 'd/newh_G_99model.pt'))  # 650d
    # modelGenerator.load_state_dict(torch.load('./result/' + t_p + 'd/newh_G_24model.pt'))  # 15d
    # modelGenerator.load_state_dict(torch.load('./result/' + t_p + 'd/newh_G_49model.pt'))  # 102d
    # modelGenerator.load_state_dict(torch.load('./result/' + t_p + 'd/newh_G_49model.pt'))  # 201d
    # modelGenerator.load_state_dict(torch.load('./result/' + t_p + 'd/newh_G_49model.pt'))  # 251d
    # modelGenerator.load_state_dict(torch.load('./result/' + t_p + 'd/newh_G_99model.pt'))  # 450d
    modelGenerator.load_state_dict(torch.load('./result/' + t_p + 'd/newh_G_74model.pt'))  # 551d

    modelGenerator.eval()

    wel = well_all[8]
    for i in range(60):
        label = Variable(Tensor(np.expand_dims(np.expand_dims(k_test[i], axis=0), axis=1)))
        fake_img = modelGenerator(wel, label)
        fake = torch.Tensor.detach(fake_img[0, 0, :, :]).cpu().numpy()

        with open('./result/'+t_p+'d/test_data/k_data_{}.plk'.format(i), 'wb') as f:
            pickle.dump(k_test[i], f)  # Save data of permeability coefficient
        with open('./result/'+t_p+'d/test_data/real_ion{}.plk'.format(i), 'wb') as f:
            pickle.dump(x_test[i], f)  #  Save uranium data from Phast
        with open('./result/'+t_p+'d/test_data/fake_ion{}.plk'.format(i), 'wb') as f:
            # pickle.dump(torch.Tensor.detach(fake_img[0, 0, :, :]).cpu().numpy(), f)  # 保存假的铀浓度
            pickle.dump(fake, f)    # save predicted uranium concentration
    print('compelet {} epoch'.format(idx))
    idx += 1