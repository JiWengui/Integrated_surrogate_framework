import matplotlib.pyplot as plt
import pickle
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
import numpy as np
from matplotlib import colors
import matplotlib as mpl

ti_chu = [12,14,18,19,21,24,55,58,44,33]    # 旧的，60个数据集之前的
ti_chu = [212,214,218,219,221,224,255,258,244,233]    # 旧的，60个数据集之前的
# ti_chu = [12,14,72,244,21,24,55,58,44,33]

time_point = ['201', '350', '551']

# 读取特定情景的图
filed_no_1 = 22+200  # 情景1         另找一个吧，22的效果不好
filed_no_1 = 32
# filed_no_2 = 45 # 情景2
all_real = [] # 顺序是201天/350天/551
all_fake = [] # 顺序是201天/350天/551
for t_p in time_point:
    with open('D:/surrogate_model/program/cdcgan_final/interpolation/result/' + t_p + 'd/test_data/real_ion{}.plk'.format(filed_no_1), 'rb') as f:
        real_1 = pickle.load(f)  # 保存铀浓度数据
    with open('D:/surrogate_model/program/cdcgan_final/interpolation/result/' + t_p + 'd/test_data/fake_ion{}.plk'.format(filed_no_1), 'rb') as f:
        fake_1 = pickle.load(f)
    all_real.append(real_1)
    all_fake.append(fake_1)


all_ssim = []   # 顺序按照上面的时间
all_mse = []
for t_p in time_point:
    ssim_single = []
    mse_single = []
    for i in range(260):
        if i not in ti_chu:
            with open('D:/surrogate_model/program/cdcgan_final/interpolation/result/' + t_p + 'd/test_data/real_ion{}.plk'.format(i), 'rb') as f:
                real = np.round(pickle.load(f), 1)  # 用来计算ssim
            with open('D:/surrogate_model/program/cdcgan_final/interpolation/result/' + t_p + 'd/test_data/fake_ion{}.plk'.format(i), 'rb') as f:
                fake = np.round(pickle.load(f), 1)
            ssim_err = ssim(real, fake, data_range=(np.max(fake) - np.min(fake)))   # 计算结构相似性SSIM
            # mse_err = mse(real, fake) # 计算均方误差MSE
            if t_p == '201' and ssim_err < 0.8:
                print('201天:{}'.format(i))
            ssim_single.append(ssim_err)
            # mse_single.append(mse_err)
    all_ssim.append(ssim_single)
    # all_mse.append(mse_single)
    # 输出SSIM和MSE区间范围和平均值
    print('第{}天 SSIM最小值为{} 最大值为{}；均值为{}'.format(t_p,np.min(ssim_single),np.max(ssim_single),np.average(ssim_single)))
    # print('第{}天 MSE最小值为{} 最大值为{}；均值为{}'.format(t_p,np.min(mse_single),np.max(mse_single),np.average(mse_single)))

vmin = 0
vmax = max(np.max(all_real), np.max(all_fake),)
norm = colors.Normalize(vmin=vmin, vmax=vmax)  # 计算色带范围

# 开始绘图
fig = plt.figure(figsize=(9, 9))
plt.rcParams['figure.dpi'] = 500    # 设置画布分辨率
plt.rc('font',family='Times New Roman')
# 图1 case1 real 102d
ax1 = fig.add_axes([.05, 0.765, .23, .23])  # 这个方法以一个包含4个值的列表作为参数来指定Axes左下角的坐标以及它的宽度和高度：[x, y, width, height]
ax1.imshow(all_real[0], norm=norm, origin='lower')
ax1.text(2, 56, "201 d", fontsize=14, color='white')
# ax1.text(2, 50, "201 d", fontsize=14, color='white')
ax1.set_xticks([], fontsize=2)
ax1.set_yticks([], fontsize=2)
ax1.set_ylabel('S',fontsize=18)
# 图2 case1 real 201d
ax2 = fig.add_axes([.315, 0.765, .23, .23])  # 这个方法以一个包含4个值的列表作为参数来指定Axes左下角的坐标以及它的宽度和高度：[x, y, width, height]
ax2.imshow(all_real[1], norm=norm, origin='lower')
ax2.text(2, 56, "350 d", fontsize=14, color='white')
# ax2.text(2, 50, "201 d", fontsize=14, color='white')
ax2.set_yticks([], fontsize=2)
ax2.set_xticks([], fontsize=2)
# # 图3 case1 real 350d
ax3 = fig.add_axes([.58, 0.765, .23, .23])  # 这个方法以一个包含4个值的列表作为参数来指定Axes左下角的坐标以及它的宽度和高度：[x, y, width, height]
ax3.imshow(all_real[2], norm=norm, origin='lower')
ax3.text(2, 56, "551 d", fontsize=14, color='white')
# ax3.text(2, 50, "350 d", fontsize=14, color='white')
ax3.set_yticks([], fontsize=2)
ax3.set_xticks([], fontsize=2)
cb = fig.add_axes([0.815, 0.765, .007, .23])
fcb2 = mpl.colorbar.ColorbarBase(norm=norm, ax=cb)
# fcb2.set_ticks([0.0, 0.2,0.4,0.6,0.8, 1.0])
# # 图4 case1 real 650d
# ax4 = fig.add_axes([.64, 0.72, .25, .25])  # 这个方法以一个包含4个值的列表作为参数来指定Axes左下角的坐标以及它的宽度和高度：[x, y, width, height]
# ax4.imshow(case1_real_ion[3], norm=norm, origin='lower')
# ax4.text(2, 56, "case 1", fontsize=14, color='white')
# ax4.text(2, 50, "650 d", fontsize=14, color='white')
# ax4.set_yticks([], fontsize=2)
# ax4.set_xticks([], fontsize=2)
# cb = fig.add_axes([0.87, 0.72, .007, .25])
# fcb2 = mpl.colorbar.ColorbarBase(norm=norm, ax=cb)
# # fcb2.set_ticks([0.0, 0.2,0.4,0.6,0.8, 1.0])
# # 图5 case1 fake 102d
ax5 = fig.add_axes([.05, 0.53, .23, .23])
ax5.imshow(all_fake[0], norm=norm, origin='lower')
ax5.set_xticks([], fontsize=2)
ax5.set_yticks([], fontsize=2)
ax5.set_ylabel('Ŝ',fontsize=18)
# # 图6 case1 fake 201d
ax6 = fig.add_axes([.315, 0.53, .23, .23])  # 这个方法以一个包含4个值的列表作为参数来指定Axes左下角的坐标以及它的宽度和高度：[x, y, width, height]
ax6.imshow(all_fake[1], norm=norm, origin='lower')
ax6.set_yticks([], fontsize=2)
ax6.set_xticks([], fontsize=2)
# # 图7 case1 fake 350d
ax7 = fig.add_axes([.58, 0.53, .23, .23])  # 这个方法以一个包含4个值的列表作为参数来指定Axes左下角的坐标以及它的宽度和高度：[x, y, width, height]
ax7.imshow(all_fake[2], norm=norm, origin='lower')
ax7.set_yticks([], fontsize=2)
ax7.set_xticks([], fontsize=2)
# # 图8 case1 fake 650d
# ax8 = fig.add_axes([.64, 0.455, .25, .25])  # 这个方法以一个包含4个值的列表作为参数来指定Axes左下角的坐标以及它的宽度和高度：[x, y, width, height]
# ax8.imshow(case1_fake_ion[3], norm=norm, origin='lower')
# ax8.set_yticks([], fontsize=2)
# ax8.set_xticks([], fontsize=2)
# # 图9 102d 残差
ax9 = fig.add_axes([.05, 0.295, .23, .23])
ax9.imshow(np.abs(all_real[0]-all_fake[0]), norm=norm, origin='lower')
ax9.set_xticks([], fontsize=2)
ax9.set_yticks([], fontsize=2)
ax9.set_ylabel('S - Ŝ',fontsize=18)

ax10 = fig.add_axes([.315, 0.295, .23, .23])
ax10.imshow(np.abs(all_real[1]-all_fake[1]), norm=norm, origin='lower')
ax10.set_xticks([], fontsize=2)
ax10.set_yticks([], fontsize=2)

ax11 = fig.add_axes([.58, 0.295, .23, .23])
ax11.imshow(np.abs(all_real[2]-all_fake[2]), norm=norm, origin='lower')
ax11.set_xticks([], fontsize=2)
ax11.set_yticks([], fontsize=2)
#

# 图1 ddim 102d
ax0 = fig.add_axes([.05, 0.055, .23, .23])   # 这个方法以一个包含4个值的列表作为参数来指定Axes左下角的坐标以及它的宽度和高度：[x, y, width, height]
ax0.hist(all_ssim[0], 80)
ax0.set_xticks([0.80,0.85,0.90,0.95,1.00], fontsize=5)
ax0.set_yticks([0,3,6,9,12,15], fontsize=5)
ax0.set_xlabel('SSIM', fontsize=13)
ax0.set_ylabel('Frequency', fontsize=13, labelpad=-1)
ax0.text(0.81,14,'201 d', fontsize=15, color='black')

ax0 = fig.add_axes([.315, 0.055, .23, .23])   # 这个方法以一个包含4个值的列表作为参数来指定Axes左下角的坐标以及它的宽度和高度：[x, y, width, height]
ax0.hist(all_ssim[1], 80)
ax0.set_xticks([0.80,0.85,0.90,0.95,1.00], fontsize=5)
ax0.set_xlabel('SSIM', fontsize=13)
# ax0.set_ylabel('Frequency', fontsize=13)
ax0.text(0.81,11.3,'350 d', fontsize=15, color='black')


ax0 = fig.add_axes([.58, 0.055, .23, .23])   # 这个方法以一个包含4个值的列表作为参数来指定Axes左下角的坐标以及它的宽度和高度：[x, y, width, height]
ax0.hist(all_ssim[2], 80)
ax0.set_xticks([0.80,0.85,0.90,0.95,1.00], fontsize=5)
ax0.set_yticks([0,2,4,6,8,10,12,14], fontsize=5)
ax0.set_xlabel('SSIM', fontsize=13)
# ax0.set_ylabel('Frequency', fontsize=13)
ax0.text(0.81,13,'551 d', fontsize=15, color='black')

plt.savefig('./residuals&ssim.png', dpi=500)
# plt.show()

