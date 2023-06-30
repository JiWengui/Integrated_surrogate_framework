import matplotlib.pyplot as plt
import pickle
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
import numpy as np

# 绘制650天的20个验证集的均值和方差
hangshu = 64
lieshu = 64
ti_chu_152 = [6,4,15,17,29,25,32,38,46,52]
# 开始先加载数据，避免每次循环都要读取一次
time_point = ['152', '300', '650']
time_point = ['300', '650']

all_real = []
all_fake = []
for t_p in time_point:
    single_day_real = []
    single_day_fake = []
    for i in range(60):
        if i not in ti_chu_152:
            with open('D:/surrogate_model/program/cdcgan_final/sensitivity/len_30/result/' + t_p + 'd/test_data/real_ion{}.plk'.format(i), 'rb') as f:
                real = pickle.load(f)  # 保存铀浓度数据
            with open('D:/surrogate_model/program/cdcgan_final/sensitivity/len_30/result/' + t_p + 'd/test_data/fake_ion{}.plk'.format(i), 'rb') as f:
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
    for i in range(2):
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


fig = plt.figure(figsize=(14, 9))
plt.rcParams['figure.dpi'] = 500  # 设置画布分辨率
plt.rc('font',family='Times New Roman')
# 图1 300 均值空间分布
ax1 = fig.add_axes([.01, 0.76, .23, .23])  # 这个方法以一个包含4个值的列表作为参数来指定Axes左下角的坐标以及它的宽度和高度：[x, y, width, height]
mean_300d_real = np.array(real_mean[0]).reshape(64, 64)  # 绘制均值场
ax1.plot([0, 64], [32, 32], color='red',linewidth=0.6)  # A-A切面
ax1.text(2, 27, 'a', fontsize=17, color='white')
ax1.text(56, 27, "a'", fontsize=17, color='white')
ax1.plot([32, 32], [0, 64], color='red',linewidth=0.6)  # B-B切面
ax1.text(26, 2, "b", fontsize=17, color='white')
ax1.text(26, 56, "b'", fontsize=17, color='white')
ax1.plot([41, 41], [0, 64], color='red',linewidth=0.6)  # C-C切面
ax1.text(43, 2, "c", fontsize=17, color='white')
ax1.text(43, 56, "c'", fontsize=17, color='white')
ax1.imshow(mean_300d_real, norm=norm_mean, origin='lower')
ax1.text(2, 55, "300 d", fontsize=17, color='white')
ax1.set_xticks([])
ax1.set_yticks([], fontsize=2)
cb = fig.add_axes([0.205, 0.76, .005, .23])
fcb1 = mpl.colorbar.ColorbarBase(norm=norm_mean, ax=cb)
fcb1.set_ticks([0.0, 0.2,0.4, 0.6, 0.8,1.0])

# 图2 300 标准差
ax1 = fig.add_axes([.20, 0.76, .23, .23])  # 这个方法以一个包含4个值的列表作为参数来指定Axes左下角的坐标以及它的宽度和高度：[x, y, width, height]
sigma_300d_real = np.array(real_sig[0]).reshape(64, 64)  # 绘制标准差场
ax1.plot([0, 64], [32, 32], color='red',linewidth=0.6)  # A-A切面
ax1.text(2, 27, 'a', fontsize=17, color='white')
ax1.text(56, 27, "a'", fontsize=17, color='white')
ax1.plot([32, 32], [0, 64], color='red',linewidth=0.6)  # B-B切面
ax1.text(26, 2, "b", fontsize=17, color='white')
ax1.text(26, 56, "b'", fontsize=17, color='white')
ax1.plot([41, 41], [0, 64], color='red',linewidth=0.6)  # C-C切面
ax1.text(43, 2, "c", fontsize=17, color='white')
ax1.text(43, 56, "c'", fontsize=17, color='white')
ax1.imshow(sigma_300d_real, norm=norm_sig, origin='lower')
ax1.text(2, 55, "300 d", fontsize=17, color='white')
ax1.set_xticks([])
ax1.set_yticks([], fontsize=2)
cb = fig.add_axes([0.395, 0.76, .005, .23])
fcb1 = mpl.colorbar.ColorbarBase(norm=norm_sig, ax=cb)
# fcb1.set_ticks([0.0, 0.2,0.4, 0.6, 0.8,1.0])

# 图3 650天均值分布
ax1 = fig.add_axes([.39, 0.76, .23, .23])
mean_650d_real = np.array(real_mean[1]).reshape(64, 64)  # 绘制均值场
ax1.plot([0, 64], [32, 32], color='red',linewidth=0.6)  # A-A切面
ax1.text(2, 27, 'a', fontsize=17, color='white')
ax1.text(56, 27, "a'", fontsize=17, color='white')
ax1.plot([32, 32], [0, 64], color='red',linewidth=0.6)  # B-B切面
ax1.text(26, 2, "b", fontsize=17, color='white')
ax1.text(26, 56, "b'", fontsize=17, color='white')
ax1.plot([41, 41], [0, 64], color='red',linewidth=0.6)  # C-C切面
ax1.text(43, 2, "c", fontsize=17, color='white')
ax1.text(43, 56, "c'", fontsize=17, color='white')
ax1.imshow(mean_650d_real, norm=norm_mean, origin='lower')
ax1.text(2, 55, "650 d", fontsize=17, color='white')
ax1.set_xticks([])
ax1.set_yticks([], fontsize=2)
cb = fig.add_axes([0.585, 0.76, .005, .23])
fcb1 = mpl.colorbar.ColorbarBase(norm=norm_mean, ax=cb)
fcb1.set_ticks([0.0, 0.2,0.4, 0.6, 0.8,1.0])

# # 图4 650天方差空间分布
ax1 = fig.add_axes([.58, 0.76, .23, .23])
sigma_650d_real = np.array(real_sig[1]).reshape(64, 64)  # 绘制标准差场
ax1.plot([0, 64], [32, 32], color='red',linewidth=0.6)  # A-A切面
ax1.text(2, 27, 'a', fontsize=17, color='white')
ax1.text(56, 27, "a'", fontsize=17, color='white')
ax1.plot([32, 32], [0, 64], color='red',linewidth=0.6)  # B-B切面
ax1.text(26, 2, "b", fontsize=17, color='white')
ax1.text(26, 56, "b'", fontsize=17, color='white')
ax1.plot([41, 41], [0, 64], color='red',linewidth=0.6)  # C-C切面
ax1.text(43, 2, "c", fontsize=17, color='white')
ax1.text(43, 56, "c'", fontsize=17, color='white')
ax1.imshow(sigma_650d_real, norm=norm_sig, origin='lower')
ax1.text(2, 55, "650 d", fontsize=17, color='white')
ax1.set_xticks([])
ax1.set_yticks([], fontsize=2)
cb = fig.add_axes([0.775, 0.76, .005, .23])
fcb1 = mpl.colorbar.ColorbarBase(norm=norm_sig, ax=cb)


'''
AA截面
'''
# # 图5 300天 均值 截面AA
ax5 = fig.add_axes([.052, 0.518, .147, .23])  # 这个方法以一个包含4个值的列表作为参数来指定Axes左下角的坐标以及它的宽度和高度：[x, y, width, height]
AA_mean_real = np.array(real_mean[0]).reshape(64, 64)[32:33, :][0]  # 取第32行的值
AA_mean_fake = np.array(fake_mean[0]).reshape(64, 64)[32:33, :][0]  # 取第32行的值
ax5.plot(np.arange(0, 64, 1), AA_mean_real, color='red',linewidth=1.0, linestyle='--', label='Phast')
ax5.plot(np.arange(0, 64, 1), AA_mean_fake, color='green',linewidth=0.6, marker='^', markersize=3,label='cDC-GAN')
ax5.legend(loc='upper right',fontsize=8.5)
ax5.set_ylabel('µ',labelpad=2,fontsize=16)
ax5.text(0.2, 0.05, "a", fontsize=17,)
ax5.text(60, 0.05, "a'", fontsize=17,)
ax5.set_xticks([])
ax5.set_yticks([0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.4])
# # 图6 300天 方差 截面AA
ax5 = fig.add_axes([.242, 0.518, .147, .23])  # 这个方法以一个包含4个值的列表作为参数来指定Axes左下角的坐标以及它的宽度和高度：[x, y, width, height]
AA_mean_real = np.array(real_sig[0]).reshape(64, 64)[32:33, :][0]  # 取第32行的值
AA_mean_fake = np.array(fake_sig[0]).reshape(64, 64)[32:33, :][0]  # 取第32行的值
ax5.plot(np.arange(0, 64, 1), AA_mean_real, color='red',linewidth=1.0, linestyle='--', label='Phast')
ax5.plot(np.arange(0, 64, 1), AA_mean_fake, color='green',linewidth=0.6, marker='^', markersize=3,label='cDC-GAN')
ax5.legend(loc='upper right',fontsize=8.5)
ax5.set_ylabel('σ',labelpad=-1,fontsize=16)
ax5.text(0.2, 0.013, "a", fontsize=17,)
ax5.text(60, 0.013, "a'", fontsize=17,)
ax5.set_xticks([])
ax5.set_yticks([0.0,0.1,0.2,0.3,0.4])
# # 图7 102天bb截面方差
ax5 = fig.add_axes([.432, 0.518, .147, .23])
AA_mean_real = np.array(real_mean[1]).reshape(64, 64)[32:33, :][0]  # 取第32行的值
AA_mean_fake = np.array(fake_mean[1]).reshape(64, 64)[32:33, :][0]  # 取第32行的值
ax5.plot(np.arange(0, 64, 1), AA_mean_real, color='red',linewidth=1.0, linestyle='--', label='Phast')
ax5.plot(np.arange(0, 64, 1), AA_mean_fake, color='green',linewidth=0.6, marker='^', markersize=3,label='cDC-GAN')
ax5.legend(loc='upper right',fontsize=8.5)
ax5.set_ylabel('µ',labelpad=2,fontsize=16)
ax5.text(0.2, 0.05, "a", fontsize=17,)
ax5.text(60, 0.05, "a'", fontsize=17,)
ax5.set_xticks([])
ax5.set_yticks([0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.4])
# # 图8 650天 标准差 AA截面
ax5 = fig.add_axes([.622, 0.518, .147, .23])  # 这个方法以一个包含4个值的列表作为参数来指定Axes左下角的坐标以及它的宽度和高度：[x, y, width, height]
AA_mean_real = np.array(real_sig[1]).reshape(64, 64)[32:33, :][0]  # 取第32行的值
AA_mean_fake = np.array(fake_sig[1]).reshape(64, 64)[32:33, :][0]  # 取第32行的值
ax5.plot(np.arange(0, 64, 1), AA_mean_real, color='red',linewidth=1.0, linestyle='--', label='Phast')
ax5.plot(np.arange(0, 64, 1), AA_mean_fake, color='green',linewidth=0.6, marker='^', markersize=3,label='cDC-GAN')
ax5.legend(loc='upper right',fontsize=8.5)
ax5.set_ylabel('σ',labelpad=-1,fontsize=16)
ax5.text(0.2, 0.013, "a", fontsize=17,)
ax5.text(60, 0.013, "a'", fontsize=17,)
ax5.set_xticks([])
ax5.set_yticks([0.0,0.1,0.2,0.3,0.4])
#
'''
BB截面
'''
# # 图9 300天 均值 截面AA
ax5 = fig.add_axes([.052, 0.275, .147, .23])  # 这个方法以一个包含4个值的列表作为参数来指定Axes左下角的坐标以及它的宽度和高度：[x, y, width, height]
BB_mean_real = np.array(real_mean[0]).reshape(64, 64)[:, 32:33].reshape(1, -1)[0] # 取第32-33行的值,31-32
BB_mean_fake = np.array(fake_mean[0]).reshape(64, 64)[:, 32:33].reshape(1, -1)[0]  # 取第32-33行的值
ax5.plot(np.arange(0, 64, 1), BB_mean_real, color='red',linewidth=1.0, linestyle='--', label='Phast')
ax5.plot(np.arange(0, 64, 1), BB_mean_fake, color='green',linewidth=0.6, marker='^', markersize=3,label='cDC-GAN')
ax5.legend(loc='upper right',fontsize=8.5)
ax5.set_ylabel('µ',labelpad=2,fontsize=16)
ax5.text(0.2, 0.03, "b", fontsize=17,)
ax5.text(60, 0.03, "b'", fontsize=17,)
ax5.set_xticks([])
ax5.set_yticks([0.0,0.2,0.4,0.6,0.8])
# # 图10 300天 方差 截面AA
ax5 = fig.add_axes([.242, 0.275, .147, .23])  # 这个方法以一个包含4个值的列表作为参数来指定Axes左下角的坐标以及它的宽度和高度：[x, y, width, height]
AA_mean_real = np.array(real_sig[0]).reshape(64, 64)[:, 32:33].reshape(1, -1)[0]  # 取第32行的值
AA_mean_fake = np.array(fake_sig[0]).reshape(64, 64)[:, 32:33].reshape(1, -1)[0]  # 取第32行的值
ax5.plot(np.arange(0, 64, 1), AA_mean_real, color='red',linewidth=1.0, linestyle='--', label='Phast')
ax5.plot(np.arange(0, 64, 1), AA_mean_fake, color='green',linewidth=0.6, marker='^', markersize=3,label='cDC-GAN')
ax5.legend(loc='upper right',fontsize=8.5)
ax5.set_ylabel('σ',labelpad=-1,fontsize=16)
ax5.text(0.2, 0.01, "b", fontsize=17,)
ax5.text(60, 0.01, "b'", fontsize=17,)
ax5.set_xticks([])
ax5.set_yticks([0.0,0.05,0.1,0.15,0.2,])
# # 图11 102天bb截面方差
ax5 = fig.add_axes([.432, 0.275, .147, .23])
AA_mean_real = np.array(real_mean[1]).reshape(64, 64)[:, 32:33].reshape(1, -1)[0]  # 取第32行的值
AA_mean_fake = np.array(fake_mean[1]).reshape(64, 64)[:, 32:33].reshape(1, -1)[0]  # 取第32行的值
ax5.plot(np.arange(0, 64, 1), AA_mean_real, color='red',linewidth=1.0, linestyle='--', label='Phast')
ax5.plot(np.arange(0, 64, 1), AA_mean_fake, color='green',linewidth=0.6, marker='^', markersize=3,label='cDC-GAN')
ax5.legend(loc='upper right',fontsize=8.5)
ax5.set_ylabel('µ',labelpad=2,fontsize=16)
ax5.text(0.2, 0.03, "b", fontsize=17,)
ax5.text(60, 0.03, "b'", fontsize=17,)
ax5.set_xticks([])
ax5.set_yticks([0.0,0.1,0.2,0.3,0.4,0.5,0.6,])
# # 图12 650天 标准差 AA截面
ax5 = fig.add_axes([.622, 0.275, .147, .23])  # 这个方法以一个包含4个值的列表作为参数来指定Axes左下角的坐标以及它的宽度和高度：[x, y, width, height]
AA_mean_real = np.array(real_sig[1]).reshape(64, 64)[:, 32:33].reshape(1, -1)[0]  # 取第32行的值
AA_mean_fake = np.array(fake_sig[1]).reshape(64, 64)[:, 32:33].reshape(1, -1)[0]  # 取第32行的值
ax5.plot(np.arange(0, 64, 1), AA_mean_real, color='red',linewidth=1.0, linestyle='--', label='Phast')
ax5.plot(np.arange(0, 64, 1), AA_mean_fake, color='green',linewidth=0.6, marker='^', markersize=3,label='cDC-GAN')
ax5.legend(loc='upper right',fontsize=8.5)
ax5.set_ylabel('σ',labelpad=-1,fontsize=16)
ax5.text(0.2, 0.01, "b", fontsize=17,)
ax5.text(60, 0.01, "b'", fontsize=17,)
ax5.set_xticks([])
ax5.set_yticks([0.0,0.05,0.1,0.15,0.2,0.25])

'''
CC截面
'''
# # 图13 300天 均值 截面AA
ax5 = fig.add_axes([.052, 0.032, .147, .23])  # 这个方法以一个包含4个值的列表作为参数来指定Axes左下角的坐标以及它的宽度和高度：[x, y, width, height]
BB_mean_real = np.array(real_mean[0]).reshape(64, 64)[:, 41:42].reshape(1, -1)[0] # 取第32-33行的值,31-32
BB_mean_fake = np.array(fake_mean[0]).reshape(64, 64)[:, 41:42].reshape(1, -1)[0]  # 取第32-33行的值
ax5.plot(np.arange(0, 64, 1), BB_mean_real, color='red',linewidth=1.0, linestyle='--', label='Phast')
ax5.plot(np.arange(0, 64, 1), BB_mean_fake, color='green',linewidth=0.6, marker='^', markersize=3,label='cDC-GAN')
ax5.legend(loc='upper right',fontsize=8.5)
ax5.set_ylabel('µ',labelpad=2,fontsize=16)
ax5.text(0.2, 0.03, "c", fontsize=17,)
ax5.text(60, 0.03, "c'", fontsize=17,)
ax5.set_xticks([0,15,30,45,60,])
ax5.set_yticks([0.0,0.2,0.4,0.6,0.8,1.0,1.2])
# # 图14 300天 方差 截面AA
ax5 = fig.add_axes([.242, 0.032, .147, .23])  # 这个方法以一个包含4个值的列表作为参数来指定Axes左下角的坐标以及它的宽度和高度：[x, y, width, height]
AA_mean_real = np.array(real_sig[0]).reshape(64, 64)[:, 41:42].reshape(1, -1)[0]  # 取第32行的值
AA_mean_fake = np.array(fake_sig[0]).reshape(64, 64)[:, 41:42].reshape(1, -1)[0]  # 取第32行的值
ax5.plot(np.arange(0, 64, 1), AA_mean_real, color='red',linewidth=1.0, linestyle='--', label='Phast')
ax5.plot(np.arange(0, 64, 1), AA_mean_fake, color='green',linewidth=0.6, marker='^', markersize=3,label='cDC-GAN')
ax5.legend(loc='upper right',fontsize=8.5)
ax5.set_ylabel('σ',labelpad=-1,fontsize=16)
ax5.text(0.2, 0.01, "c", fontsize=17,)
ax5.text(60, 0.01, "c'", fontsize=17,)
ax5.set_xticks([0,15,30,45,60,])
ax5.set_yticks([0.0,0.1,0.2,0.3,0.4])
# # 图15 102天bb截面方差
ax5 = fig.add_axes([.432, 0.032, .147, .23])
AA_mean_real = np.array(real_mean[1]).reshape(64, 64)[:, 41:42].reshape(1, -1)[0]  # 取第32行的值
AA_mean_fake = np.array(fake_mean[1]).reshape(64, 64)[:, 41:42].reshape(1, -1)[0] # 取第32行的值
ax5.plot(np.arange(0, 64, 1), AA_mean_real, color='red',linewidth=1.0, linestyle='--', label='Phast')
ax5.plot(np.arange(0, 64, 1), AA_mean_fake, color='green',linewidth=0.6, marker='^', markersize=3,label='cDC-GAN')
ax5.legend(loc='upper right',fontsize=8.5)
ax5.set_ylabel('µ',labelpad=2,fontsize=16)
ax5.text(0.2, 0.03, "c", fontsize=17,)
ax5.text(60, 0.03, "c'", fontsize=17,)
ax5.set_xticks([0,15,30,45,60,])
ax5.set_yticks([0.0,0.2,0.4,0.6,0.8,1.0,1.2])
# # 图16 650天 标准差 AA截面
ax5 = fig.add_axes([.622, 0.032, .147, .23])  # 这个方法以一个包含4个值的列表作为参数来指定Axes左下角的坐标以及它的宽度和高度：[x, y, width, height]
AA_mean_real = np.array(real_sig[1]).reshape(64, 64)[:, 41:42].reshape(1, -1)[0]  # 取第32行的值
AA_mean_fake = np.array(fake_sig[1]).reshape(64, 64)[:, 41:42].reshape(1, -1)[0]  # 取第32行的值
ax5.plot(np.arange(0, 64, 1), AA_mean_real, color='red',linewidth=1.0, linestyle='--', label='Phast')
ax5.plot(np.arange(0, 64, 1), AA_mean_fake, color='green',linewidth=0.6, marker='^', markersize=3,label='cDC-GAN')
ax5.legend(loc='upper right',fontsize=8.5)
ax5.set_ylabel('σ',labelpad=-1,fontsize=16)
ax5.text(0.2, 0.01, "c", fontsize=17,)
ax5.text(60, 0.01, "c'", fontsize=17,)
ax5.set_xticks([0,15,30,45,60,])
ax5.set_yticks([0.0,0.1,0.2,0.3,0.4])

# plt.show()
plt.savefig('./len_30_cross-sections.png', dpi=500)
plt.close()

