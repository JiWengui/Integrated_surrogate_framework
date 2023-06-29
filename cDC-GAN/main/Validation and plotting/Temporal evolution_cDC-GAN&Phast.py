import matplotlib.pyplot as plt
import pickle
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
import numpy as np

# 绘制650天的20个验证集的均值和方差
hangshu = 64
lieshu = 64
# 开始先加载数据，避免每次循环都要读取一次
time_point = ['15','102', '152','201','251','300','350','450','551','650']

real = []
fake = []
for t_p in time_point:
    with open('D:/surrogate_model/program/cdcgan_final/main/result/' + t_p + 'd/test_data/real_ion22.plk', 'rb') as f:
        real_one = pickle.load(f)  # 保存铀浓度数据
    with open('D:/surrogate_model/program/cdcgan_final/main/result/' + t_p + 'd/test_data/fake_ion22.plk', 'rb') as f:
        fake_one = pickle.load(f)
    real.append(real_one)
    fake.append(fake_one)

from matplotlib import colors
import matplotlib as mpl
vmin = 0
vmax = max(np.max(real), np.max(fake))
norm = colors.Normalize(vmin=vmin, vmax=vmax)  # 计算色带范围


fig = plt.figure(figsize=(14, 9))
plt.rcParams['figure.dpi'] = 500  # 设置画布分辨率
plt.rc('font',family='Times New Roman') # 将字体设置为times new roman

# 图1 real 15
ax1 = fig.add_axes([.02, 0.76, .23, .23])  # 这个方法以一个包含4个值的列表作为参数来指定Axes左下角的坐标以及它的宽度和高度：[x, y, width, height]
ax1.imshow(real[0], norm=norm, origin='lower')
ax1.text(2, 55, "15 d", fontsize=15, color='white')
ax1.text(47, 55, "Phast", fontsize=15, color='white')
ax1.set_xticks([])
ax1.set_yticks([0, 15, 30, 45, 60], fontsize=2)
# 图2 real 102
ax2 = fig.add_axes([.175, 0.76, .23, .23])
ax2.imshow(real[1], norm=norm, origin='lower')
ax2.text(2, 55, "102 d", fontsize=15, color='white')
ax2.text(47, 55, "Phast", fontsize=15, color='white')
ax2.set_xticks([])
ax2.set_yticks([])
# 图3 real 152
ax3 = fig.add_axes([.33, 0.76, .23, .23])
ax3.imshow(real[2], norm=norm, origin='lower')
ax3.text(2, 55, "152 d", fontsize=15, color='white')
ax3.text(47, 55, "Phast", fontsize=15, color='white')
ax3.set_xticks([])
ax3.set_yticks([])
# 图4 real 201
ax4 = fig.add_axes([.485, 0.76, .23, .23])
ax4.imshow(real[3], norm=norm, origin='lower')
ax4.text(2, 55, "201 d", fontsize=15, color='white')
ax4.text(47, 55, "Phast", fontsize=15, color='white')
ax4.set_xticks([])
ax4.set_yticks([])
# 图5 real 251
ax5 = fig.add_axes([.64, 0.76, .23, .23])  # 这个方法以一个包含4个值的列表作为参数来指定Axes左下角的坐标以及它的宽度和高度：[x, y, width, height]
ax5.imshow(real[4], norm=norm, origin='lower')
ax5.text(2, 55, "251 d", fontsize=15, color='white')
ax5.text(47, 55, "Phast", fontsize=15, color='white')
ax5.set_xticks([])
ax5.set_yticks([])
# 颜色柱
cb = fig.add_axes([0.835, 0.76, .007, .23])
fcb1 = mpl.colorbar.ColorbarBase(norm=norm, ax=cb)
fcb1.set_ticks([0.0, 0.2,0.4, 0.6, 0.8,1.0])
# 图6 real 300
ax6 = fig.add_axes([.02,0.52, .23, .23])
ax6.imshow(real[5], norm=norm, origin='lower')
ax6.text(2, 55, "300 d", fontsize=15, color='white')
ax6.text(47, 55, "Phast", fontsize=15, color='white')
ax6.set_xticks([])
ax6.set_yticks([0, 15, 30, 45, 60], fontsize=2)
# 图7 real 350
ax7 = fig.add_axes([.175, 0.52, .23, .23])
ax7.imshow(real[6], norm=norm, origin='lower')
ax7.text(2, 55, "350 d", fontsize=15, color='white')
ax7.text(47, 55, "Phast", fontsize=15, color='white')
ax7.set_xticks([])
ax7.set_yticks([])
# 图8 real 450
ax8 = fig.add_axes([.33, 0.52, .23, .23])
ax8.imshow(real[7], norm=norm, origin='lower')
ax8.text(2, 55, "450 d", fontsize=15, color='white')
ax8.text(47, 55, "Phast", fontsize=15, color='white')
ax8.set_xticks([])
ax8.set_yticks([])
# 图4 real 551
ax9 = fig.add_axes([.485, 0.52, .23, .23])
ax9.imshow(real[8], norm=norm, origin='lower')
ax9.text(2, 55, "551 d", fontsize=15, color='white')
ax9.text(47, 55, "Phast", fontsize=15, color='white')
ax9.set_xticks([])
ax9.set_yticks([])
# 图5 real 650
ax10 = fig.add_axes([.64, 0.52, .23, .23])  # 这个方法以一个包含4个值的列表作为参数来指定Axes左下角的坐标以及它的宽度和高度：[x, y, width, height]
ax10.imshow(real[9], norm=norm, origin='lower')
ax10.text(2, 55, "650 d", fontsize=15, color='white')
ax10.text(47, 55, "Phast", fontsize=15, color='white')
ax10.set_xticks([])
ax10.set_yticks([])


'''
FAKE
'''
# 图11 fake 15
ax11 = fig.add_axes([.02, 0.28, .23, .23])
ax11.imshow(fake[0], norm=norm, origin='lower')
ax11.text(2, 55, "15 d", fontsize=15, color='white')
ax11.text(34, 55, "cDC-GAN", fontsize=15, color='white')
ax11.set_xticks([])
ax11.set_yticks([0, 15, 30, 45, 60], fontsize=2)
# 图12 fake 102
ax12 = fig.add_axes([.175, 0.28, .23, .23])
ax12.imshow(fake[1], norm=norm, origin='lower')
ax12.text(2, 55, "102 d", fontsize=15, color='white')
ax12.text(34, 55, "cDC-GAN", fontsize=15, color='white')
ax12.set_xticks([])
ax12.set_yticks([])
# 图13 fake 152
ax13 = fig.add_axes([.33, 0.28, .23, .23])
ax13.imshow(fake[2], norm=norm, origin='lower')
ax13.text(2, 55, "152 d", fontsize=15, color='white')
ax13.text(34, 55, "cDC-GAN", fontsize=15, color='white')
ax13.set_xticks([])
ax13.set_yticks([])
# 图14 fake 201
ax14 = fig.add_axes([.485, 0.28, .23, .23])
ax14.imshow(fake[3], norm=norm, origin='lower')
ax14.text(2, 55, "201 d", fontsize=15, color='white')
ax14.text(34, 55, "cDC-GAN", fontsize=15, color='white')
ax14.set_xticks([])
ax14.set_yticks([])
# 图15 fake 251
ax15 = fig.add_axes([.64, 0.28, .23, .23])  # 这个方法以一个包含4个值的列表作为参数来指定Axes左下角的坐标以及它的宽度和高度：[x, y, width, height]
ax15.imshow(fake[4], norm=norm, origin='lower')
ax15.text(2, 55, "251 d", fontsize=15, color='white')
ax15.text(34, 55, "cDC-GAN", fontsize=15, color='white')
ax15.set_xticks([])
ax15.set_yticks([])
# 图16 fake 300
ax16 = fig.add_axes([.02, 0.04, .23, .23])
ax16.imshow(real[5], norm=norm, origin='lower')
ax16.text(2, 55, "300 d", fontsize=15, color='white')
ax16.text(34, 55, "cDC-GAN", fontsize=15, color='white')
ax16.set_xticks([0, 15, 30, 45, 60], fontsize=2)
ax16.set_yticks([0, 15, 30, 45, 60], fontsize=2)
# 图17 fake 350
ax17 = fig.add_axes([.175, 0.04, .23, .23])
ax17.imshow(fake[6], norm=norm, origin='lower')
ax17.text(2, 55, "350d", fontsize=15, color='white')
ax17.text(34, 55, "cDC-GAN", fontsize=15, color='white')
ax17.set_xticks([0, 15, 30, 45, 60], fontsize=2)
ax17.set_yticks([])
# 图18 fake 450
ax18 = fig.add_axes([.33, 0.04, .23, .23])
ax18.imshow(fake[7], norm=norm, origin='lower')
ax18.text(2, 55, "450d", fontsize=15, color='white')
ax18.text(34, 55, "cDC-GAN", fontsize=15, color='white')
ax18.set_xticks([0, 15, 30, 45, 60], fontsize=2)
ax18.set_yticks([])
# 图19 fake 551
ax19 = fig.add_axes([.485, 0.04, .23, .23])
ax19.imshow(fake[8], norm=norm, origin='lower')
ax19.text(2, 55, "551d", fontsize=15, color='white')
ax19.text(34, 55, "cDC-GAN", fontsize=15, color='white')
ax19.set_xticks([0, 15, 30, 45, 60], fontsize=2)
ax19.set_yticks([])
# 图20 fake 650
ax20 = fig.add_axes([.64, 0.04, .23, .23])  # 这个方法以一个包含4个值的列表作为参数来指定Axes左下角的坐标以及它的宽度和高度：[x, y, width, height]
ax20.imshow(fake[9], norm=norm, origin='lower')
ax20.text(2, 55, "650d", fontsize=15, color='white')
ax20.text(34, 55, "cDC-GAN", fontsize=15, color='white')
ax20.set_xticks([0, 15, 30, 45, 60], fontsize=2)
ax20.set_yticks([])

# plt.show()
plt.savefig('./temporal evolution.png', dpi=500)
plt.close()

