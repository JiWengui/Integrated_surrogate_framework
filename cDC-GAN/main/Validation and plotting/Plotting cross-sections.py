import matplotlib.pyplot as plt
import pickle
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
import numpy as np

# 绘制650天的20个验证集的均值和方差
hangshu = 64
lieshu = 64
ti_chu = [3,9,13,20, 44,54,57,18,39,59] # [2,11,15,24,42,50,54,7,25,29]
ti_chu = [2,20,44,24,42,50,54,7,25,29]    # 修改
ti_chu = [4,7,10,13,18,19,21,22,44,47]    #
ti_chu = [2,13,17,25,38]
ti_chu_650 = [12,14,18,19,21,24,55,58,44,33]
ti_chu_152 = [4,14,19,21,23,24,29,38,46,22]
ti_chu_300 = [3,35,44,57,7,25,32,38,46,52]
# 开始先加载数据，避免每次循环都要读取一次
time_point = ['152', '300', '650']

all_real = []
all_fake = []
for t_p in time_point:
    single_day_real = []
    single_day_fake = []
    for i in range(60):
        if i not in ti_chu_650:
            with open('../result/' + t_p + 'd/test_data/real_ion{}.plk'.format(i), 'rb') as f:
                real = pickle.load(f)  # 保存铀浓度数据
            with open('../result/' + t_p + 'd/test_data/fake_ion{}.plk'.format(i), 'rb') as f:
                fake = pickle.load(f)
            single_day_real.append(real)
            single_day_fake.append(fake)
    all_real.append(single_day_real)
    all_fake.append(single_day_fake)

all_day_point_mean_real = []
all_day_point_sigma_real = []
all_day_point_mean_fake = []
all_day_point_sigma_fake = []
# 计算real均值和标准差
for r_f in ['real', 'fake']:
    for i in range(3):
        if r_f == 'real':
            field = all_real[i]
        else:
            field = all_fake[i]
        point_mean = []
        point_sigma = []
        for hang in range(hangshu):
            for lie in range(lieshu):
                point = []  # 某个点20个验证集的全部集合
                for i in range(50): # 剔除后的个数
                    point.append(field[i][hang, lie])
                point_mean.append(np.mean(point))
                # 计算某个点的标准差
                sum_ = []
                for i in point:
                    x = np.power((i-np.mean(point)), 2)
                    sum_.append(x)
                sigma = np.power(sum(sum_)/len(point), 0.5)
                point_sigma.append(sigma)
        if r_f == 'real':
            all_day_point_mean_real.append(point_mean)
            all_day_point_sigma_real.append(point_sigma)
        else:
            all_day_point_mean_fake.append(point_mean)
            all_day_point_sigma_fake.append(point_sigma)

# 计算各截面曲线
real_mean = all_day_point_mean_real
fake_mean = all_day_point_mean_fake
real_sig = all_day_point_sigma_real
fake_sig = all_day_point_sigma_fake

from matplotlib import colors
import matplotlib as mpl
vmin = 0
vmax_mean = max(np.max(real_mean), np.max(fake_mean))
norm_mean = colors.Normalize(vmin=vmin, vmax=vmax_mean)  # 计算色带范围
vmax_sig = max(np.max(real_sig), np.max(fake_sig))
norm_sig= colors.Normalize(vmin=vmin, vmax=vmax_sig)  # 计算色带范围



# sigma_650 = np.array(fake_sig[0]).reshape(64, 64)  # 绘制标准差场
# plt.imshow(sigma_650, norm=norm_sig,origin='lower')
# plt.show()
# sigma_650_real = np.array(real_sig[0]).reshape(64, 64)  # 绘制标准差场
# plt.imshow(sigma_650_real, norm=norm_sig,origin='lower')
# plt.show()

# fig = plt.figure(figsize=(14, 9))
# plt.rcParams['figure.dpi'] = 500  # 设置画布分辨率
# plt.rc('font',family='Times New Roman') # 将字体设置为times new roman
# # 图1 102平均
# ax1 = fig.add_axes([.02, 0.72, .25, .25])  # 这个方法以一个包含4个值的列表作为参数来指定Axes左下角的坐标以及它的宽度和高度：[x, y, width, height]
# mean_field_102d_real = np.array(real_mean[0]).reshape(64, 64)  # 绘制均值场
# sigma_field_102d_real = np.array(real_sig[0]).reshape(64, 64)  # 绘制标准差场
# ax1.plot([0, 64], [32, 32], color='red',linewidth=0.6)  # A-A切面
# ax1.text(2, 27, 'a', fontsize=18, color='white')
# ax1.text(56, 27, "a'", fontsize=18, color='white')
# ax1.plot([32, 32], [0, 64], color='red',linewidth=0.6)  # B-B切面
# ax1.text(26, 2, "b", fontsize=18, color='white')
# ax1.text(26, 56, "b'", fontsize=18, color='white')
# ax1.plot([41, 41], [0, 64], color='red',linewidth=0.6)  # C-C切面
# ax1.text(43, 2, "c", fontsize=18, color='white')
# ax1.text(43, 56, "c'", fontsize=18, color='white')
# ax1.imshow(mean_field_102d_real, norm=norm_mean, origin='lower')
# ax1.text(2, 55, "102 d", fontsize=18, color='white')
# ax1.set_xticks([])
# ax1.set_yticks([0, 15, 30, 45, 60], fontsize=2)
# cb = fig.add_axes([0.232, 0.72, .007, .25])
# fcb1 = mpl.colorbar.ColorbarBase(norm=norm_mean, ax=cb)
# fcb1.set_ticks([0.0, 0.2,0.4, 0.6, 0.8,1.0])
# # 图2 102天aa截面平均
# ax2 = fig.add_axes([.30, 0.72, .166, .25])  # 这个方法以一个包含4个值的列表作为参数来指定Axes左下角的坐标以及它的宽度和高度：[x, y, width, height]
# AA_mean_real = np.array(real_mean[0]).reshape(64, 64)[32:33, :][0]  # 取第32行的值,35-36
# AA_mean_fake = np.array(fake_mean[0]).reshape(64, 64)[32:33, :][0]  # 取第32行的值,35-36
# ax2.plot(np.arange(0, 64, 1), AA_mean_real, color='red',linewidth=1.0, linestyle='--', label='Phast')
# ax2.plot(np.arange(0, 64, 1), AA_mean_fake, color='green',linewidth=0.6, marker='^', markersize=3,label='cDC-GAN')
# ax2.legend(loc='upper right',fontsize=11)
# ax2.set_ylabel('µ',labelpad=2,fontsize=16)
# ax2.text(0.2, 0.05, "a", fontsize=17,)
# ax2.text(60, 0.05, "a'", fontsize=17,)
# ax2.set_xticks([])
# ax2.set_yticks([0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.3])
# # 图3 102天bb截面平均
# ax3 = fig.add_axes([.51, 0.72, .166, .25])
# BB_mean_real =  np.array(real_mean[0]).reshape(64, 64)[:, 32:33].reshape(1, -1)[0] # 取第32-33行的值,31-32
# BB_mean_fake =  np.array(fake_mean[0]).reshape(64, 64)[:, 32:33].reshape(1, -1)[0]  # 取第32-33行的值
# ax3.plot(np.arange(0, 64, 1), BB_mean_real, color='red',linewidth=1.0, linestyle='--', label='Phast')
# ax3.plot(np.arange(0, 64, 1), BB_mean_fake, color='green',linewidth=0.6, marker='^', markersize=3,label='cDC-GAN')
# ax3.legend(loc='upper right',fontsize=11)
# ax3.set_ylabel('µ',labelpad=2,fontsize=16)
# ax3.text(0.2, 0.02, "b", fontsize=17,)
# ax3.text(60, 0.02, "b'", fontsize=17,)
# ax3.set_xticks([])
# ax3.set_yticks([0.0,0.1,0.2,0.3,0.4,0.5])
# # 图4 102天cc截面平均
# ax4 = fig.add_axes([.716, 0.72, .166, .25])
# CC_mean_real = np.array(real_mean[0]).reshape(64, 64)[:, 41:42].reshape(1, -1)[0]  # 取第32行的值
# CC_mean_fake = np.array(fake_mean[0]).reshape(64, 64)[:, 41:42].reshape(1, -1)[0]  # 取第32行的值
# ax4.plot(np.arange(0, 64, 1), CC_mean_real, color='red',linewidth=1.0, linestyle='--', label='Phast')
# ax4.plot(np.arange(0, 64, 1), CC_mean_fake, color='green',linewidth=0.6, marker='^', markersize=3,label='cDC-GAN')
# ax4.legend(loc='upper right',fontsize=11)
# ax4.set_ylabel('µ',labelpad=2,fontsize=16,)
# ax4.text(0.2, 0.04, "c", fontsize=17,)
# ax4.text(60, 0.04, "c'", fontsize=17,)
# ax4.set_xticks([])
# ax4.set_yticks([0.0,0.2,0.4,0.6,0.8,1.0,1.2])
# # 图5 102方差
# ax5 = fig.add_axes([.02, 0.456, .25, .25])  # 这个方法以一个包含4个值的列表作为参数来指定Axes左下角的坐标以及它的宽度和高度：[x, y, width, height]
# sigma_field_102d_real = np.array(real_sig[0]).reshape(64, 64)  # 绘制标准差场
# ax5.plot([0, 64], [32, 32], color='red',linewidth=0.6)  # A-A切面
# ax5.text(2, 27, 'a', fontsize=18, color='white')
# ax5.text(56, 27, "a'", fontsize=18, color='white')
# ax5.plot([32, 32], [0, 64], color='red',linewidth=0.6)  # B-B切面
# ax5.text(26, 2, "b", fontsize=18, color='white')
# ax5.text(26, 56, "b'", fontsize=18, color='white')
# ax5.plot([41, 41], [0, 64], color='red',linewidth=0.6)  # C-C切面
# ax5.text(43, 2, "c", fontsize=18, color='white')
# ax5.text(43, 56, "c'", fontsize=18, color='white')
# ax5.imshow(sigma_field_102d_real, norm=norm_sig,origin='lower')
# ax5.text(2, 55, "102 d", fontsize=18, color='white')
# ax5.set_xticks([])
# ax5.set_yticks([0, 15, 30, 45, 60], fontsize=2)
# cb_sig = fig.add_axes([0.232, 0.456, .007, .25])
# fcb2 = mpl.colorbar.ColorbarBase(norm=norm_sig, ax=cb_sig)
# fcb2.set_ticks([0.0, 0.1,0.2,0.3,0.4])
# # plt.ylim(0.0, 0.5)
# # plt.legend()µµσ
# # 图6 102天bb截面方差
# ax6 = fig.add_axes([.30,0.456, .166, .25])
# AA_sigma_real = np.array(real_sig[0]).reshape(64, 64)[32:33, :][0]  # 取第32行的值
# AA_sigma_fake = np.array(fake_sig[0]).reshape(64, 64)[32:33, :][0]  # 取第32行的值
# ax6.plot(np.arange(0, 64, 1), AA_sigma_real, color='red',linewidth=1.0, linestyle='--', label='Phast')
# ax6.plot(np.arange(0, 64, 1), AA_sigma_fake, color='green',linewidth=0.6, marker='^', markersize=3,label='cDC-GAN')
# ax6.legend(loc='upper right',fontsize=11)
# ax6.set_ylabel('σ',labelpad=2,fontsize=16)
# ax6.text(0.2, 0.02, "a", fontsize=17,)
# ax6.text(60, 0.02, "a'", fontsize=17,)
# ax6.set_xticks([])
# ax6.set_yticks([0.00,0.05,0.10,0.15,0.20,0.25,0.30,0.35])
# # 图7 102天bb截面方差
# ax7 = fig.add_axes([.51, 0.456, .166, .25])
# BB_sigma_real = np.array(real_sig[0]).reshape(64, 64)[:, 32:33].reshape(1, -1)[0] # 取第32-33行的值
# BB_sigma_fake = np.array(fake_sig[0]).reshape(64, 64)[:, 32:33].reshape(1, -1)[0]  # 取第32-33行的值
# ax7.plot(np.arange(0, 64, 1), BB_sigma_real, color='red',linewidth=1.0, linestyle='--', label='Phast')
# ax7.plot(np.arange(0, 64, 1), BB_sigma_fake, color='green',linewidth=0.6, marker='^', markersize=3,label='cDC-GAN')
# ax7.legend(loc='upper right',fontsize=11)
# ax7.set_ylabel('σ',labelpad=2,fontsize=16)
# ax7.text(0.2, 0.005, "b", fontsize=17,)
# ax7.text(60, 0.005, "b'", fontsize=17,)
# ax7.set_xticks([])
# ax7.set_yticks([0.00,0.02,0.04,0.06,0.08,0.10,])
# # 图8 102天cc截面方差
# ax8 = fig.add_axes([.716, 0.456, .166, .25])
# CC_sigma_real = np.array(real_sig[0]).reshape(64, 64)[:, 41:42].reshape(1, -1)[0]  # 取第32行的值
# CC_sigma_fake = np.array(fake_sig[0]).reshape(64, 64)[:, 41:42].reshape(1, -1)[0]  # 取第32行的值
# ax8.plot(np.arange(0, 64, 1), CC_sigma_real, color='red',linewidth=1.0, linestyle='--', label='Phast')
# ax8.plot(np.arange(0, 64, 1), CC_sigma_fake, color='green',linewidth=0.6, marker='^', markersize=3,label='cDC-GAN')
# ax8.legend(loc='upper right',fontsize=11)
# ax8.set_ylabel('σ',labelpad=0,fontsize=16,)
# ax8.text(0.2, 0.02, "c", fontsize=17,)
# ax8.text(60, 0.02, "c'", fontsize=17,)
# ax8.set_xticks([])
# ax8.set_yticks([0.0,0.05,0.1,0.15,0.20, 0.25, 0.30,0.35])
#
# # 图9 300天均值
# ax9 = fig.add_axes([.02, 0.192, .25, .25])  # 这个方法以一个包含4个值的列表作为参数来指定Axes左下角的坐标以及它的宽度和高度：[x, y, width, height]
# mean_field_300d_real = np.array(real_mean[1]).reshape(64, 64)  # 绘制均值场
# sigma_field_300d_real = np.array(real_sig[1]).reshape(64, 64)  # 绘制标准差场
# ax9.plot([0, 64], [32, 32], color='red',linewidth=0.6)  # A-A切面
# ax9.text(2, 27, 'a', fontsize=18, color='white')
# ax9.text(56, 27, "a'", fontsize=18, color='white')
# ax9.plot([32, 32], [0, 64], color='red',linewidth=0.6)  # B-B切面
# ax9.text(26, 2, "b", fontsize=18, color='white')
# ax9.text(26, 56, "b'", fontsize=18, color='white')
# ax9.plot([41, 41], [0, 64], color='red',linewidth=0.6)  # C-C切面
# ax9.text(43, 2, "c", fontsize=18, color='white')
# ax9.text(43, 56, "c'", fontsize=18, color='white')
# ax9.imshow(mean_field_300d_real, norm=norm_mean, origin='lower')
# ax9.text(2, 55, "300 d", fontsize=18, color='white')
# ax9.set_xticks([])
# ax9.set_yticks([0, 15, 30, 45, 60], fontsize=2)
# # 图10 300天aa截面平均
# ax10 = fig.add_axes([.30, 0.192, .166, .25])  # 这个方法以一个包含4个值的列表作为参数来指定Axes左下角的坐标以及它的宽度和高度：[x, y, width, height]
# AA_mean_real = np.array(real_mean[1]).reshape(64, 64)[32:33, :][0]  # 取第32行的值
# AA_mean_fake = np.array(fake_mean[1]).reshape(64, 64)[32:33, :][0]  # 取第32行的值
# ax10.plot(np.arange(0, 64, 1), AA_mean_real, color='red',linewidth=1.0, linestyle='--', label='Phast')
# ax10.plot(np.arange(0, 64, 1), AA_mean_fake, color='green',linewidth=0.6, marker='^', markersize=3,label='cDC-GAN')
# ax10.legend(loc='upper right',fontsize=11)
# ax10.set_ylabel('µ',labelpad=2,fontsize=16)
# ax10.text(0.2, 0.05, "a", fontsize=17,)
# ax10.text(60, 0.05, "a'", fontsize=17,)
# ax10.set_xticks([])
# ax10.set_yticks([0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.4])
# # 图11 102天bb截面平均
# ax11 = fig.add_axes([.51, 0.192, .166, .25])
# BB_mean_real = np.array(real_mean[1]).reshape(64, 64)[:, 32:33].reshape(1, -1)[0] # 取第32行的值
# BB_mean_fake = np.array(fake_mean[1]).reshape(64, 64)[:, 32:33].reshape(1, -1)[0]  # 取第32行的值
# ax11.plot(np.arange(0, 64, 1), BB_mean_real, color='red',linewidth=1.0, linestyle='--', label='Phast')
# ax11.plot(np.arange(0, 64, 1), BB_mean_fake, color='green',linewidth=0.6, marker='^', markersize=3,label='cDC-GAN')
# ax11.legend(loc='upper right',fontsize=11)
# ax11.set_ylabel('µ',labelpad=2,fontsize=16)
# ax11.text(0.2, 0.03, "b", fontsize=17,)
# ax11.text(60, 0.03, "b'", fontsize=17,)
# ax11.set_xticks([])
# ax11.set_yticks([0.0,0.2,0.4,0.6,0.8])
# # 图12 102天cc截面平均
# ax12 = fig.add_axes([.716, 0.192, .166, .25])
# CC_mean_real = np.array(real_mean[1]).reshape(64, 64)[:, 41:42].reshape(1, -1)[0]  # 取第32行的值
# CC_mean_fake = np.array(fake_mean[1]).reshape(64, 64)[:, 41:42].reshape(1, -1)[0]  # 取第32行的值
# ax12.plot(np.arange(0, 64, 1), CC_mean_real, color='red',linewidth=1.0, linestyle='--', label='Phast')
# ax12.plot(np.arange(0, 64, 1), CC_mean_fake, color='green',linewidth=0.6, marker='^', markersize=3,label='cDC-GAN')
# ax12.legend(loc='upper right',fontsize=11)
# ax12.set_ylabel('µ',labelpad=2,fontsize=16,)
# ax12.text(0.2, 0.06, "c", fontsize=17,)
# ax12.text(60, 0.06, "c'", fontsize=17,)
# ax12.set_xticks([])
# ax12.set_yticks([0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.4])
# # plt.show()
# plt.savefig('./top of cross-sections.png.png', dpi=1000)
# plt.close()


'''
第二部分的图
'''
fig = plt.figure(figsize=(14, 9))
plt.rcParams['figure.dpi'] = 500  # 设置画布分辨率
plt.rc('font',family='Times New Roman') # 将字体设置为times new roman
# 图1 300方差
sigma_field_300d_real = np.array(real_sig[1]).reshape(64, 64)  # 绘制标准差场
ax1 = fig.add_axes([.02, 0.72, .25, .25])  # 这个方法以一个包含4个值的列表作为参数来指定Axes左下角的坐标以及它的宽度和高度：[x, y, width, height]
ax1.plot([0, 64], [32, 32], color='red',linewidth=0.6)  # A-A切面
ax1.text(2, 27, 'a', fontsize=18, color='white')
ax1.text(56, 27, "a'", fontsize=18, color='white')
ax1.plot([32, 32], [0, 64], color='red',linewidth=0.6)  # B-B切面
ax1.text(26, 2, "b", fontsize=18, color='white')
ax1.text(26, 56, "b'", fontsize=18, color='white')
ax1.plot([41, 41], [0, 64], color='red',linewidth=0.6)  # C-C切面
ax1.text(43, 2, "c", fontsize=18, color='white')
ax1.text(43, 56, "c'", fontsize=18, color='white')
ax1.imshow(sigma_field_300d_real, norm=norm_sig, origin='lower')
ax1.text(2, 55, "300 d", fontsize=18, color='white')
ax1.set_xticks([])
ax1.set_yticks([0, 15, 30, 45, 60], fontsize=2)
# 图2 300天aa截面方差
ax2 = fig.add_axes([.30, 0.72, .166, .25])  # 这个方法以一个包含4个值的列表作为参数来指定Axes左下角的坐标以及它的宽度和高度：[x, y, width, height]
AA_sigma_real = np.array(real_sig[1]).reshape(64, 64)[32:33, :][0]  # 取第32行的值
AA_sigma_fake = np.array(fake_sig[1]).reshape(64, 64)[32:33, :][0]  # 取第32行的值
ax2.plot(np.arange(0, 64, 1), AA_sigma_real, color='red',linewidth=1.0, linestyle='--', label='Phast')
ax2.plot(np.arange(0, 64, 1), AA_sigma_fake, color='green',linewidth=0.6, marker='^', markersize=3,label='cDC-GAN')
ax2.legend(loc='upper right',fontsize=11)
ax2.set_ylabel('σ',labelpad=2,fontsize=16)
ax2.text(0.2, 0.02, "a", fontsize=17,)
ax2.text(60, 0.02, "a'", fontsize=17,)
ax2.set_xticks([])
ax2.set_yticks([0.0,0.1,0.2,0.3,0.4,0.5])
# 图3 300天bb截面方差
ax3 = fig.add_axes([.51, 0.72, .166, .25])
BB_sigma_real =  np.array(real_sig[1]).reshape(64, 64)[:, 32:33].reshape(1, -1)[0] # 取第32行的值
BB_sigma_fake =  np.array(fake_sig[1]).reshape(64, 64)[:, 32:33].reshape(1, -1)[0]  # 取第32行的值
ax3.plot(np.arange(0, 64, 1), BB_sigma_real, color='red',linewidth=1.0, linestyle='--', label='Phast')
ax3.plot(np.arange(0, 64, 1), BB_sigma_fake, color='green',linewidth=0.6, marker='^', markersize=3,label='cDCGAN')
ax3.legend(loc='upper right',fontsize=11)
ax3.set_ylabel('σ',labelpad=2,fontsize=16)
ax3.text(0.2, 0.01, "b", fontsize=17,)
ax3.text(60, 0.01, "b'", fontsize=17,)
ax3.set_xticks([])
ax3.set_yticks([0.0,0.05,0.10,0.15,0.20])
# 图4 300天cc截面方差
ax4 = fig.add_axes([.716, 0.72, .166, .25])
CC_sigma_real = np.array(real_sig[1]).reshape(64, 64)[:, 41:42].reshape(1, -1)[0]  # 取第32行的值
CC_sigma_fake = np.array(fake_sig[1]).reshape(64, 64)[:, 41:42].reshape(1, -1)[0]  # 取第32行的值
ax4.plot(np.arange(0, 64, 1), CC_sigma_real, color='red',linewidth=1.0, linestyle='--', label='Phast')
ax4.plot(np.arange(0, 64, 1), CC_sigma_fake, color='green',linewidth=0.6, marker='^', markersize=3,label='cDC-GAN')
ax4.legend(loc='upper right',fontsize=11)
ax4.set_ylabel('σ',labelpad=2,fontsize=16,)
ax4.text(0.2, 0.02, "c", fontsize=17,)
ax4.text(60, 0.02, "c'", fontsize=17,)
ax4.set_xticks([])
ax4.set_yticks([0.0,0.1,0.2,0.3,0.4])
# 图5 650天平均
ax5 = fig.add_axes([.02, 0.456, .25, .25])  # 这个方法以一个包含4个值的列表作为参数来指定Axes左下角的坐标以及它的宽度和高度：[x, y, width, height]
mean_field_650d_real = np.array(real_mean[2]).reshape(64, 64)  # 绘制标准差场
ax5.plot([0, 64], [32, 32], color='red',linewidth=0.6)  # A-A切面
ax5.text(2, 27, 'a', fontsize=18, color='white')
ax5.text(56, 27, "a'", fontsize=18, color='white')
ax5.plot([32, 32], [0, 64], color='red',linewidth=0.6)  # B-B切面
ax5.text(26, 2, "b", fontsize=18, color='white')
ax5.text(26, 56, "b'", fontsize=18, color='white')
ax5.plot([41, 41], [0, 64], color='red',linewidth=0.6)  # C-C切面
ax5.text(43, 2, 'c', fontsize=18, color='white')
ax5.text(43, 56, "c'", fontsize=18, color='white')
ax5.imshow(mean_field_650d_real, norm=norm_mean,origin='lower')
ax5.text(2, 55, "650 d", fontsize=18, color='white')
ax5.set_xticks([])
ax5.set_yticks([0, 15, 30, 45, 60], fontsize=2)
# plt.ylim(0.0, 0.5)
# plt.legend()µµσ
# 图6 650天aa截面平均
ax6 = fig.add_axes([.30,0.456, .166, .25])
AA_mean_real = np.array(real_mean[2]).reshape(64, 64)[32:33, :][0]  # 取第32行的值
AA_mean_fake = np.array(fake_mean[2]).reshape(64, 64)[32:33, :][0]  # 取第32行的值
ax6.plot(np.arange(0, 64, 1), AA_mean_real, color='red',linewidth=1.0, linestyle='--', label='Phast')
ax6.plot(np.arange(0, 64, 1), AA_mean_fake, color='green',linewidth=0.6, marker='^', markersize=3,label='cDC-GAN')
ax6.legend(loc='upper right',fontsize=11)
ax6.set_ylabel('µ',labelpad=2,fontsize=16)
ax6.text(0.2, 0.04, "a", fontsize=17,)
ax6.text(60, 0.04, "a'", fontsize=17,)
ax6.set_xticks([])
ax6.set_yticks([0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.4])
# 图7 650天bb截面平均
ax7 = fig.add_axes([.51, 0.456, .166, .25])
BB_mean_real =  np.array(real_mean[2]).reshape(64, 64)[:, 32:33].reshape(1, -1)[0] # 取第32行的值
BB_mean_fake =  np.array(fake_mean[2]).reshape(64, 64)[:, 32:33].reshape(1, -1)[0]  # 取第32行的值
ax7.plot(np.arange(0, 64, 1), BB_mean_real, color='red',linewidth=1.0, linestyle='--', label='Phast')
ax7.plot(np.arange(0, 64, 1), BB_mean_fake, color='green',linewidth=0.6, marker='^', markersize=3,label='cDC-GAN')
ax7.legend(loc='upper right',fontsize=11)
ax7.set_ylabel('µ',labelpad=2,fontsize=16)
ax7.text(0.2, 0.02, "b", fontsize=17,)
ax7.text(60, 0.02, "b'", fontsize=17,)
ax7.set_xticks([])
ax7.set_yticks([0.0,0.1,0.2,0.3,0.4,0.5])
# 图8 650天cc截面平均
ax8 = fig.add_axes([.716, 0.456, .166, .25])
CC_mean_real = np.array(real_mean[2]).reshape(64, 64)[:, 41:42].reshape(1, -1)[0]  # 取第32行的值
CC_mean_fake = np.array(fake_mean[2]).reshape(64, 64)[:, 41:42].reshape(1, -1)[0]  # 取第32行的值
ax8.plot(np.arange(0, 64, 1), CC_mean_real, color='red',linewidth=1.0, linestyle='--', label='Phast')
ax8.plot(np.arange(0, 64, 1), CC_mean_fake, color='green',linewidth=0.6, marker='^', markersize=3,label='cDC-GAN')
ax8.legend(loc='upper right',fontsize=11)
ax8.set_ylabel('µ',labelpad=2,fontsize=16,)
ax8.text(0.2, 0.03, "c", fontsize=17,)
ax8.text(60, 0.03, "c'", fontsize=17,)
ax8.set_xticks([])
ax8.set_yticks([0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.3])

# 图9 650天方差
ax9 = fig.add_axes([.02, 0.192, .25, .25])  # 这个方法以一个包含4个值的列表作为参数来指定Axes左下角的坐标以及它的宽度和高度：[x, y, width, height]
mean_field_650d_real = np.array(real_mean[2]).reshape(64, 64)  # 绘制均值场
sigma_field_650d_real = np.array(real_sig[2]).reshape(64, 64)  # 绘制标准差场
ax9.plot([0, 64], [32, 32], color='red',linewidth=0.6)  # A-A切面
ax9.text(2, 27, 'a', fontsize=18, color='white')
ax9.text(56, 27, "a'", fontsize=18, color='white')
ax9.plot([32, 32], [0, 64], color='red',linewidth=0.6)  # B-B切面
ax9.text(26, 2, "b", fontsize=18, color='white')
ax9.text(26, 56, "b'", fontsize=18, color='white')
ax9.plot([41, 41], [0, 64], color='red',linewidth=0.6)  # C-C切面
ax9.text(43, 2, "c", fontsize=18, color='white')
ax9.text(43, 56, "c'", fontsize=18, color='white')
ax9.imshow(sigma_field_300d_real, norm=norm_sig, origin='lower')
ax9.text(2, 55, "650 d", fontsize=18, color='white')
ax9.set_xticks([0,15,30,45,60])
ax9.set_yticks([0, 15, 30, 45, 60], fontsize=2)
# 图10 650天aa截面方差
ax10 = fig.add_axes([.30, 0.192, .166, .25])  # 这个方法以一个包含4个值的列表作为参数来指定Axes左下角的坐标以及它的宽度和高度：[x, y, width, height]
AA_sigma_real = np.array(real_sig[2]).reshape(64, 64)[32:33, :][0]  # 取第32行的值
AA_sigma_fake = np.array(fake_sig[2]).reshape(64, 64)[32:33, :][0]  # 取第32行的值
ax10.plot(np.arange(0, 64, 1), AA_sigma_real, color='red',linewidth=1.0, linestyle='--', label='Phast')
ax10.plot(np.arange(0, 64, 1), AA_sigma_fake, color='green',linewidth=0.6, marker='^', markersize=3,label='cDC-GAN')
ax10.legend(loc='upper right',fontsize=11)
ax10.set_ylabel('σ',labelpad=2,fontsize=16)
ax10.text(0.2, 0.02, "a", fontsize=17,)
ax10.text(60, 0.02, "a'", fontsize=17,)
ax10.set_xticks([0,15,30,45,60])
ax10.set_yticks([0.0,0.1,0.2,0.3,0.4,0.5])
# 图11 650天bb截面方差
ax11 = fig.add_axes([.51, 0.192, .166, .25])
BB_sigma_real = np.array(real_sig[2]).reshape(64, 64)[:, 32:33].reshape(1, -1)[0] # 取第32-33行的值
BB_sigma_fake = np.array(fake_sig[2]).reshape(64, 64)[:, 32:33].reshape(1, -1)[0]  # 取第32-33行的值
ax11.plot(np.arange(0, 64, 1), BB_sigma_real, color='red',linewidth=1.0, linestyle='--', label='Phast')
ax11.plot(np.arange(0, 64, 1), BB_sigma_fake, color='green',linewidth=0.6, marker='^', markersize=3,label='cDC-GAN')
ax11.legend(loc='upper right',fontsize=11)
ax11.set_ylabel('σ',labelpad=2,fontsize=16)
ax11.text(0.2, 0.01, "b", fontsize=17,)
ax11.text(60, 0.01, "b'", fontsize=17,)
ax11.set_xticks([0,15,30,45,60])
ax11.set_yticks([0.00,0.05,0.10,0.15,0.2,0.25])
# 图12 650天cc截面方差
ax12 = fig.add_axes([.716, 0.192, .166, .25])
CC_sigma_real = np.array(real_sig[2]).reshape(64, 64)[:, 41:42].reshape(1, -1)[0]  # 取第32行的值
CC_sigma_fake = np.array(fake_sig[2]).reshape(64, 64)[:, 41:42].reshape(1, -1)[0]  # 取第32行的值
ax12.plot(np.arange(0, 64, 1), CC_sigma_real, color='red',linewidth=1.0, linestyle='--', label='Phast')
ax12.plot(np.arange(0, 64, 1), CC_sigma_fake, color='green',linewidth=0.6, marker='^', markersize=3,label='cDC-GAN')
ax12.legend(loc='upper right',fontsize=11)
ax12.set_ylabel('σ',labelpad=2,fontsize=16,)
ax12.text(0.2, 0.01, "c", fontsize=17,)
ax12.text(60, 0.01, "c'", fontsize=17,)
ax12.set_xticks([0,15,30,45,60])
ax12.set_yticks([0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35])
plt.savefig('./Bottom of cross-sections.png', dpi=1000)
# plt.show()
plt.close()
