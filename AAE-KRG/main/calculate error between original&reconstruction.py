
import argparse
import os
import numpy as np
import pandas as pd
import pickle
import itertools
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt

parser.add_argument("--n_epochs", type=int, default=5000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent code")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
args = parser.parse_args()
print(args)
cuda = torch.cuda.is_available()
img_shape = (args.channels, args.img_size, args.img_size)
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(64*64, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, args.latent_dim)
        )
    def forward(self, img):
        img_flat = img[0,0,:,:].reshape(1, -1)
        x = self.model(img_flat)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(args.latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 64*64),
            nn.Tanh(),
        )
    def forward(self, z):
        img_flat = self.model(z)
        img = img_flat.view(img_flat.shape[0], *img_shape)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(args.latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
    def forward(self, z):
        validity = self.model(z)
        return validity

# AE
encoder = Encoder()
decoder = Decoder()
# 2) discriminator
discriminator = Discriminator()
# loss
adversarial_loss = nn.BCELoss()
reconstruction_loss = nn.MSELoss()
# optimizer
optimizer_G = torch.optim.Adam(itertools.chain(encoder.parameters(), decoder.parameters()), lr=args.lr,
                               betas=(args.b1, args.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
if cuda:
    encoder.cuda()
    decoder.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    reconstruction_loss.cuda()

k_input = []
ti_chu = [42,80,99,126,155,173,176,192,237,240,]
for m in range(260):
    if m not in ti_chu:
        k_data = np.array(pd.read_csv('./data/k_data/第' + str(m) + '个场.txt',header=None, sep=' '))
        max = np.max(k_data)
        min = np.min(k_data)
        k_normal = (k_data-min)/(max-min)
        k_input.append(k_normal)

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
import torch
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
encoder = Encoder()
decoder = Decoder()
encoder.cuda()
decoder.cuda()
# load trained encoder and decoder
encoder.load_state_dict(torch.load('./result/encoder_1999model.pt'))
decoder.load_state_dict(torch.load("./result/decoder_1999model.pt"))
# Set to verification status
encoder.eval()
decoder.eval()
# Start Verification
all_fake = []
for i in range(250):
    x = Variable(Tensor(np.expand_dims(np.expand_dims(k_input[i], axis=0), axis=1)))
    laten = encoder(x)
    fake_img = decoder(laten)
    all_fake.append(fake_img)
all_ssim = []
all_mse = []
for idx, k_real in enumerate(k_input):
    k_real = np.round(k_real, 1)
    k_fake = np.round(torch.Tensor.detach(all_fake[idx][0, 0,:,:]).cpu().numpy(), 1)
    ssim_err = ssim(k_real, k_fake, data_range=(np.max(k_fake) - np.min(k_fake)))   # caculate SSIM
    mse_err = mse(k_real, k_fake) # caculate MSE
    all_ssim.append(ssim_err)
    all_mse.append(mse_err)

# plotting
fig = plt.figure(figsize=(8, 6))
plt.rcParams['figure.dpi'] = 500
plt.rc('font',family='Times New Roman')
# subplot 1, ssim
ax0 = fig.add_axes([.08, 0.3, 0.4, .4])
ax0.hist(all_ssim, 50)
ax0.set_xticks([0.65,0.75,0.85,0.95])
ax0.set_ylabel('Frequency', fontsize=11, labelpad=0)
ax0.set_xlabel('SSIM', fontsize=10)
# subfiguer 2, mse
ax1 = fig.add_axes([.57, 0.3, .4, .4])
ax1.hist(all_mse, 50)
ax1.set_ylabel('Frequency', fontsize=11, labelpad=0)
ax1.set_xlabel('MSE', fontsize=10)
plt.savefig('./SSIM&MSE.png',dpi=500)
# plt.show()
