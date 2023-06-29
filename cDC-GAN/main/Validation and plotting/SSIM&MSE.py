import matplotlib.pyplot as plt
import pickle
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
import numpy as np

# ti_chu = [3,9,13,20, 44,54,57,18,39,59]
# ti_chu = [2,11,15,24,42,50,54,7,25,29]    # 之前
ti_chu = [2,20,44,24,42,50,54,7,25,29]    # 修改
ti_chu = [12,14,18,19,21,24,55,58,44,33]

time_point = ['102','201','300','450','650']
all_ssim = []   # 顺序按照上面的时间
all_mse = []
for t_p in time_point:
    # 读取102天真假数据，计算SSIM和MSE
    ssim_single = []
    mse_single = []
    for i in range(60):
        if i not in ti_chu:
            with open('D:/surrogate_model/program/cdcgan_final/main/result/' + t_p + 'd/test_data/real_ion{}.plk'.format(i), 'rb') as f:
                real = np.round(pickle.load(f), 1)  # 保存铀浓度数据
            with open('D:/surrogate_model/program/cdcgan_final/main/result/' + t_p + 'd/test_data/fake_ion{}.plk'.format(i), 'rb') as f:
                fake = np.round(pickle.load(f), 1)
            ssim_err = ssim(real, fake, data_range=(np.max(fake) - np.min(fake)))   # 计算结构相似性SSIM
            mse_err = mse(real, fake) # 计算均方误差MSE
            if t_p == '650' and ssim_err < 0.85:
                print('650天:{}'.format(i))
            ssim_single.append(ssim_err)
            mse_single.append(mse_err)
    all_ssim.append(ssim_single)
    all_mse.append(mse_single)
    # 输出SSIM和MSE区间范围和平均值
    print('第{}天 SSIM最小值为{} 最大值为{}；均值为{}'.format(t_p,np.min(ssim_single),np.max(ssim_single),np.average(ssim_single)))
    print('第{}天 MSE最小值为{} 最大值为{}；均值为{}'.format(t_p,np.min(mse_single),np.max(mse_single),np.average(mse_single)))


# 开始绘图
fig = plt.figure(figsize=(14, 7))
plt.rcParams['figure.dpi'] = 500    # 设置画布分辨率
plt.rc('font',family='Times New Roman')
# 图1 ddim 102d
ax0 = fig.add_axes([.03, 0.65, .15, .33])   # 这个方法以一个包含4个值的列表作为参数来指定Axes左下角的坐标以及它的宽度和高度：[x, y, width, height]
ax0.hist(all_ssim[0], 30)
ax0.set_xticks([0.80,0.85,0.90,0.95,1.00], fontsize=5)
ax0.set_xlabel('SSIM', fontsize=11)
ax0.set_ylabel('Frequency', fontsize=11)
ax0.text(0.81,4.7,'102 d', fontsize=14, color='black')
# 图2 ssim 201d
ax1 = fig.add_axes([.205, 0.65, .15, .33])   # 这个方法以一个包含4个值的列表作为参数来指定Axes左下角的坐标以及它的宽度和高度：[x, y, width, height]
ax1.hist(all_ssim[1], 30)
ax1.set_xticks([0.80,0.85,0.90,0.95,1.00], fontsize=5)
ax1.set_xlabel('SSIM', fontsize=11)
# ax1.set_ylabel('Frequency', fontsize=12)
ax1.text(0.81,8.34,'201 d', fontsize=14, color='black')
# 图3 ssim 300d
ax2 = fig.add_axes([.38, 0.65, .15, .33])
ax2.hist(all_ssim[2], 30)
ax2.set_xticks([0.80,0.85,0.90,0.95,1.00], fontsize=5)
ax2.set_xlabel('SSIM', fontsize=11)
# ax2.set_ylabel('Frequency', fontsize=12)
ax2.text(0.81,6.55,'300 d', fontsize=14, color='black')
# 图4 ssim 450d
ax3 = fig.add_axes([.555, 0.65, .15, .33])
ax3.hist(all_ssim[3], 30)
ax3.set_xticks([0.80,0.85,0.90,0.95,1.00], fontsize=5)
ax3.set_xlabel('SSIM', fontsize=11)
# ax3.set_ylabel('Frequency', fontsize=12)
ax3.text(0.81,6.3,'450 d', fontsize=14, color='black')
ax3.set_yticks([0,1,2,3,4,5,6,7])
# 图5 ssim 650d
ax4 = fig.add_axes([.73, 0.65, .15, .33])
ax4.hist(all_ssim[4], 30)
ax4.set_xticks([0.80,0.85,0.90,0.95,1.00], fontsize=5)
ax4.set_xlabel('SSIM', fontsize=11)
# ax4.set_ylabel('Frequency', fontsize=12)
ax4.text(0.81,5.65,'650 d', fontsize=14, color='black')
ax4.set_yticks([0,1,2,3,4,5,6])

# 图6 mse 102d
ax5 = fig.add_axes([.03, 0.23, .15, .33])
ax5.hist(all_mse[0], 30)
ax5.set_xlabel('MSE', fontsize=11)
ax5.set_ylabel('Frequency', fontsize=11, labelpad=-1)
# 图7 mse 201d
ax6 = fig.add_axes([.205, 0.23, .15, .33])   # 这个方法以一个包含4个值的列表作为参数来指定Axes左下角的坐标以及它的宽度和高度：[x, y, width, height]
ax6.hist(all_mse[1], 30)
ax6.set_xlabel('MSE', fontsize=11)
# 图8 mse 300d
ax7 = fig.add_axes([.38, 0.23, .15, .33])
ax7.hist(all_mse[2], 30)
ax7.set_xlabel('MSE', fontsize=11)
# 图9 mse 450d
ax9 = fig.add_axes([.555, 0.23, .15, .33])
ax9.hist(all_mse[3], 30)
ax9.set_xlabel('MSE', fontsize=10)
# 图10 mse 650d
ax10 = fig.add_axes([.73, 0.23, .15, .33])
ax10.hist(all_mse[4], 30)
ax10.set_xlabel('MSE', fontsize=11)

plt.savefig('./SSIM&MSE.png', dpi=500)
# plt.show()

