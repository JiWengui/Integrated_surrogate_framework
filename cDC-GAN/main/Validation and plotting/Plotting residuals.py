
import matplotlib.pyplot as plt
from matplotlib import colors
import pickle
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
import matplotlib as mpl

time_point = ['102', '201','350', '650']
filed_no_1 = 22  # 情景1
filed_no_2 = 45 # 情景2

case1_real_ion = [] # 顺序是102天/201天/350天/650天
case1_fake_ion = [] # 顺序是102天/201天/350天/650天
case2_real_ion = [] # 顺序是102天/201天/350天/650天
case2_fake_ion = [] # 顺序是102天/201天/350天/650天
for t_p in time_point:
    with open('D:/surrogate_model/program/cdcgan_final/main/result/' + t_p + 'd/test_data/real_ion{}.plk'.format(filed_no_1), 'rb') as f:
        real_1 = pickle.load(f)  # 保存铀浓度数据
    with open('D:/surrogate_model/program/cdcgan_final/main/result/' + t_p + 'd/test_data/fake_ion{}.plk'.format(filed_no_1), 'rb') as f:
        fake_1 = pickle.load(f)
    with open('D:/surrogate_model/program/cdcgan_final/main/result/' + t_p + 'd/test_data/real_ion{}.plk'.format(filed_no_2), 'rb') as f:
        real_2 = pickle.load(f)  # 保存铀浓度数据
    with open('D:/surrogate_model/program/cdcgan_final/main/result/' + t_p + 'd/test_data/fake_ion{}.plk'.format(filed_no_2), 'rb') as f:
        fake_2 = pickle.load(f)
    case1_real_ion.append(real_1)
    case1_fake_ion.append(fake_1)
    case2_real_ion.append(real_2)
    case2_fake_ion.append(fake_2)

vmin = 0
vmax = max(np.max(case1_real_ion), np.max(case1_fake_ion), np.max(case2_real_ion), np.max(case2_fake_ion))
norm = colors.Normalize(vmin=vmin, vmax=vmax)  # 计算色带范围

# '''作神经网络结构示意图的图'''
# plt.imshow(case2_real_ion[2], norm=norm, origin='lower')    # 650天真
# plt.savefig('F:/神经网络作图/real.png', dpi=500)
# plt.imshow(case2_fake_ion[2], norm=norm, origin='lower')    # 650天假
# plt.savefig('F:/神经网络作图/fake.png', dpi=500)
# plt.imshow(k_2, norm=norm, origin='lower')  # 渗透系数图
# plt.savefig('F:/神经网络作图/k.png', dpi=500)

fig = plt.figure(figsize=(12, 9))
plt.rcParams['figure.dpi'] = 1000  # 设置画布分辨率
plt.rc('font',family='Times New Roman') # 将字体设置为times new roman
# 图1 case1 real 102d
ax1 = fig.add_axes([.04, 0.72, .25, .25])  # 这个方法以一个包含4个值的列表作为参数来指定Axes左下角的坐标以及它的宽度和高度：[x, y, width, height]
ax1.imshow(case1_real_ion[0], norm=norm, origin='lower')
ax1.text(2, 56, "case 1", fontsize=16, color='white')
ax1.text(2, 50, "102 d", fontsize=16, color='white')
ax1.set_xticks([], fontsize=2)
ax1.set_yticks([], fontsize=2)
ax1.set_ylabel('S',fontsize=18)
# 图2 case1 real 201d
ax2 = fig.add_axes([.24, 0.72, .25, .25])  # 这个方法以一个包含4个值的列表作为参数来指定Axes左下角的坐标以及它的宽度和高度：[x, y, width, height]
ax2.imshow(case1_real_ion[1], norm=norm, origin='lower')
ax2.text(2, 56, "case 1", fontsize=16, color='white')
ax2.text(2, 50, "201 d", fontsize=16, color='white')
ax2.set_yticks([], fontsize=2)
ax2.set_xticks([], fontsize=2)
# 图3 case1 real 350d
ax3 = fig.add_axes([.44, 0.72, .25, .25])  # 这个方法以一个包含4个值的列表作为参数来指定Axes左下角的坐标以及它的宽度和高度：[x, y, width, height]
ax3.imshow(case1_real_ion[2], norm=norm, origin='lower')
ax3.text(2, 56, "case 1", fontsize=16, color='white')
ax3.text(2, 50, "350 d", fontsize=16, color='white')
ax3.set_yticks([], fontsize=2)
ax3.set_xticks([], fontsize=2)
# 图4 case1 real 650d
ax4 = fig.add_axes([.64, 0.72, .25, .25])  # 这个方法以一个包含4个值的列表作为参数来指定Axes左下角的坐标以及它的宽度和高度：[x, y, width, height]
ax4.imshow(case1_real_ion[3], norm=norm, origin='lower')
ax4.text(2, 56, "case 1", fontsize=16, color='white')
ax4.text(2, 50, "650 d", fontsize=16, color='white')
ax4.set_yticks([], fontsize=2)
ax4.set_xticks([], fontsize=2)
cb = fig.add_axes([0.87, 0.72, .007, .25])
fcb2 = mpl.colorbar.ColorbarBase(norm=norm, ax=cb)
# fcb2.set_ticks([0.0, 0.2,0.4,0.6,0.8, 1.0])
# 图5 case1 fake 102d
ax5 = fig.add_axes([.04, 0.455, .25, .25])
ax5.imshow(case1_fake_ion[0], norm=norm, origin='lower')
ax5.set_xticks([], fontsize=2)
ax5.set_yticks([], fontsize=2)
ax5.set_ylabel('Ŝ',fontsize=18)
# 图6 case1 fake 201d
ax6 = fig.add_axes([.24, 0.455, .25, .25])  # 这个方法以一个包含4个值的列表作为参数来指定Axes左下角的坐标以及它的宽度和高度：[x, y, width, height]
ax6.imshow(case1_fake_ion[1], norm=norm, origin='lower')
ax6.set_yticks([], fontsize=2)
ax6.set_xticks([], fontsize=2)
# 图7 case1 fake 350d
ax7 = fig.add_axes([.44, 0.455, .25, .25])  # 这个方法以一个包含4个值的列表作为参数来指定Axes左下角的坐标以及它的宽度和高度：[x, y, width, height]
ax7.imshow(case1_fake_ion[2], norm=norm, origin='lower')
ax7.set_yticks([], fontsize=2)
ax7.set_xticks([], fontsize=2)
# 图8 case1 fake 650d
ax8 = fig.add_axes([.64, 0.455, .25, .25])  # 这个方法以一个包含4个值的列表作为参数来指定Axes左下角的坐标以及它的宽度和高度：[x, y, width, height]
ax8.imshow(case1_fake_ion[3], norm=norm, origin='lower')
ax8.set_yticks([], fontsize=2)
ax8.set_xticks([], fontsize=2)
# 图9 102d 残差
ax9 = fig.add_axes([.04, 0.190, .25, .25])
ax9.imshow(np.abs(case1_real_ion[0]-case1_fake_ion[0]), norm=norm, origin='lower')
ax9.set_xticks([], fontsize=2)
ax9.set_yticks([], fontsize=2)
ax9.set_ylabel('S - Ŝ',fontsize=18)
# 图10 102天aa截面平均
ax10 = fig.add_axes([.24, 0.19, .25, .25])  # 这个方法以一个包含4个值的列表作为参数来指定Axes左下角的坐标以及它的宽度和高度：[x, y, width, height]
ax10.imshow(np.abs(case1_real_ion[1]-case1_fake_ion[1]), norm=norm, origin='lower')
ax10.set_xticks([], fontsize=2)
ax10.set_yticks([], fontsize=2)
# 图11 102天aa截面平均
ax11 = fig.add_axes([.44, 0.19, .25, .25])  # 这个方法以一个包含4个值的列表作为参数来指定Axes左下角的坐标以及它的宽度和高度：[x, y, width, height]
ax11.imshow(np.abs(case1_real_ion[2]-case1_fake_ion[2]), norm=norm, origin='lower')
ax11.set_xticks([], fontsize=2)
ax11.set_yticks([], fontsize=2)
# 图8 102天aa截面平均
ax12 = fig.add_axes([.64, 0.19, .25, .25])  # 这个方法以一个包含4个值的列表作为参数来指定Axes左下角的坐标以及它的宽度和高度：[x, y, width, height]
ax12.imshow(np.abs(case1_real_ion[2]-case1_fake_ion[2]), norm=norm, origin='lower')
ax12.set_xticks([], fontsize=2)
ax12.set_yticks([], fontsize=2)
plt.savefig('./Top of the residual figure.png', dpi=1000)
# plt.show()
plt.close()

'''
第二部分的图
'''
fig = plt.figure(figsize=(12, 9))
plt.rcParams['figure.dpi'] = 1000  # 设置画布分辨率
plt.rc('font',family='Times New Roman') # 将字体设置为times new roman
# 图1 case1 real 102d
ax1 = fig.add_axes([.04, 0.72, .25, .25])  # 这个方法以一个包含4个值的列表作为参数来指定Axes左下角的坐标以及它的宽度和高度：[x, y, width, height]
ax1.imshow(case2_real_ion[0], norm=norm, origin='lower')
ax1.text(2, 56, "case 2", fontsize=16, color='white')
ax1.text(2, 50, "102 d", fontsize=16, color='white')
ax1.set_xticks([], fontsize=2)
ax1.set_yticks([], fontsize=2)
ax1.set_ylabel('S',fontsize=18)
# 图2 case1 real 201d
ax2 = fig.add_axes([.24, 0.72, .25, .25])  # 这个方法以一个包含4个值的列表作为参数来指定Axes左下角的坐标以及它的宽度和高度：[x, y, width, height]
ax2.imshow(case2_real_ion[1], norm=norm, origin='lower')
ax2.text(2, 56, "case 2", fontsize=16, color='white')
ax2.text(2, 50, "201 d", fontsize=16, color='white')
ax2.set_yticks([], fontsize=2)
ax2.set_xticks([], fontsize=2)
# 图3 case1 real 350d
ax3 = fig.add_axes([.44, 0.72, .25, .25])  # 这个方法以一个包含4个值的列表作为参数来指定Axes左下角的坐标以及它的宽度和高度：[x, y, width, height]
ax3.imshow(case2_real_ion[2], norm=norm, origin='lower')
ax3.text(2, 56, "case 2", fontsize=16, color='white')
ax3.text(2, 50, "350 d", fontsize=16, color='white')
ax3.set_yticks([], fontsize=2)
ax3.set_xticks([], fontsize=2)
# 图4 case1 real 650d
ax4 = fig.add_axes([.64, 0.72, .25, .25])  # 这个方法以一个包含4个值的列表作为参数来指定Axes左下角的坐标以及它的宽度和高度：[x, y, width, height]
ax4.imshow(case2_real_ion[3], norm=norm, origin='lower')
ax4.text(2, 56, "case 2", fontsize=16, color='white')
ax4.text(2, 50, "650 d", fontsize=16, color='white')
ax4.set_yticks([], fontsize=2)
ax4.set_xticks([], fontsize=2)
# cb = fig.add_axes([0.87, 0.72, .007, .25])
# fcb2 = mpl.colorbar.ColorbarBase(norm=norm, ax=cb)
# fcb2.set_ticks([0.0, 0.2,0.4,0.6,0.8, 1.0])
# 图5 case1 fake 102d
ax5 = fig.add_axes([.04, 0.455, .25, .25])
ax5.imshow(case2_fake_ion[0], norm=norm, origin='lower')
ax5.set_xticks([], fontsize=2)
ax5.set_yticks([], fontsize=2)
ax5.set_ylabel('Ŝ',fontsize=18)
# 图6 case1 fake 201d
ax6 = fig.add_axes([.24, 0.455, .25, .25])  # 这个方法以一个包含4个值的列表作为参数来指定Axes左下角的坐标以及它的宽度和高度：[x, y, width, height]
ax6.imshow(case2_fake_ion[1], norm=norm, origin='lower')
ax6.set_yticks([], fontsize=2)
ax6.set_xticks([], fontsize=2)
# 图7 case1 fake 350d
ax7 = fig.add_axes([.44, 0.455, .25, .25])  # 这个方法以一个包含4个值的列表作为参数来指定Axes左下角的坐标以及它的宽度和高度：[x, y, width, height]
ax7.imshow(case2_fake_ion[2], norm=norm, origin='lower')
ax7.set_yticks([], fontsize=2)
ax7.set_xticks([], fontsize=2)
# 图8 case1 fake 650d
ax8 = fig.add_axes([.64, 0.455, .25, .25])  # 这个方法以一个包含4个值的列表作为参数来指定Axes左下角的坐标以及它的宽度和高度：[x, y, width, height]
ax8.imshow(case2_fake_ion[3], norm=norm, origin='lower')
ax8.set_yticks([], fontsize=2)
ax8.set_xticks([], fontsize=2)
# 图9 102d 残差
ax9 = fig.add_axes([.04, 0.190, .25, .25])
ax9.imshow(np.abs(case2_real_ion[0]-case2_fake_ion[0]), norm=norm, origin='lower')
ax9.set_xticks([], fontsize=2)
ax9.set_yticks([], fontsize=2)
ax9.set_ylabel('S - Ŝ',fontsize=18)
# 图10 102天aa截面平均
ax10 = fig.add_axes([.24, 0.19, .25, .25])  # 这个方法以一个包含4个值的列表作为参数来指定Axes左下角的坐标以及它的宽度和高度：[x, y, width, height]
ax10.imshow(np.abs(case2_real_ion[1]-case2_fake_ion[1]), norm=norm, origin='lower')
ax10.set_xticks([], fontsize=2)
ax10.set_yticks([], fontsize=2)
# 图11 102天aa截面平均
ax11 = fig.add_axes([.44, 0.19, .25, .25])  # 这个方法以一个包含4个值的列表作为参数来指定Axes左下角的坐标以及它的宽度和高度：[x, y, width, height]
ax11.imshow(np.abs(case2_real_ion[2]-case2_fake_ion[2]), norm=norm, origin='lower')
ax11.set_xticks([], fontsize=2)
ax11.set_yticks([], fontsize=2)
# 图8 102天aa截面平均
ax12 = fig.add_axes([.64, 0.19, .25, .25])  # 这个方法以一个包含4个值的列表作为参数来指定Axes左下角的坐标以及它的宽度和高度：[x, y, width, height]
ax12.imshow(np.abs(case2_real_ion[2]-case2_fake_ion[2]), norm=norm, origin='lower')
ax12.set_xticks([], fontsize=2)
ax12.set_yticks([], fontsize=2)
plt.savefig('./Bottom of the residual figure.png', dpi=1000)
# # plt.show()
plt.close()

