import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
from matplotlib import colors
import torch
import matplotlib as mpl

'''
Load data for plotting uncertainty interval
'''
ens_no = 260
start_num = 0
path_txt_9 = '../all_data/well_data/var_0.45/well_9'    # 'txt' file save path of C1
path_txt_10 = '../all_data/well_data/var_0.45/well_10'   # 'txt' file save path of C1
well_9 = []
well_10 = []
ti_chu = [219]
for i in range(251):
    if i not in ti_chu:
        df_random_9 = pd.read_csv(path_txt_9+'/no_'+str(i+start_num)+'_9号井所有化学成分.txt', sep=' ')
        df_random_10 = pd.read_csv(path_txt_10+'/no_'+str(i+start_num)+'_10号井所有化学成分.txt', sep=' ')
        X = df_random_9['Time']
        wel_9_sig = np.array(df_random_9['U'])
        wel_10_sig = np.array(df_random_10['U'])
        well_9.append(wel_9_sig)
        well_10.append(wel_10_sig)

# load KRG model of C1
with open('../all_data/krg_model/var_0.45/KRG_wel9_model.pkl', 'rb') as f:
    sm = pickle.load(f)
all_laten = []
for i in range(251):
    if i not in ti_chu:
        with open('../all_data/k_laten/var_0.45/no_{}'.format(i), 'rb') as f:
            laten = pickle.load(f)
            all_laten.append(torch.Tensor.detach(laten).cpu().numpy())
# predict uranium concentration curve by trained KRG
k_laten_test = np.array(all_laten[200:250], dtype='float64')[:,0,:]
well_9_test = np.array(well_9[200:250], dtype='float64')
well_10_test = np.array(well_10[200:250])
y_test = sm.predict_values(k_laten_test)
# Filling the space between curves
linear_max_sm9 = []
linear_min_sm9 = []
linear_max_w9 = []
linear_min_w9 = []
for j in range(157):
    temp_sm = []
    temp_w9 = []
    for i in range(50):
        temp_sm.append(y_test[i][j])
        temp_w9.append(well_9_test[i][j])
    linear_max_sm9.append(np.max(temp_sm))
    linear_min_sm9.append(np.min(temp_sm))
    linear_max_w9.append(np.max(temp_w9))
    linear_min_w9.append(np.min(temp_w9))

# load KRG model of C2
with open('../all_data/krg_model/var_0.45/KRG_wel10_model.pkl', 'rb') as f:
    sm = pickle.load(f)
all_laten = []
for i in range(251):
    if i not in ti_chu:
        with open('../all_data/k_laten/var_0.45/no_{}'.format(i), 'rb') as f:
            laten = pickle.load(f)
            all_laten.append(torch.Tensor.detach(laten).cpu().numpy())

k_laten_test = np.array(all_laten[200:250], dtype='float64')[:,0,:]
well_9_test = np.array(well_9[200:250], dtype='float64')
well_10_test = np.array(well_10[200:250])
y_test = sm.predict_values(k_laten_test)
linear_max_sm10 = []
linear_min_sm10 = []
linear_max_w10 = []
linear_min_w10 = []
for j in range(157):
    temp_sm = []
    temp_w10 = []
    for i in range(50):
        temp_sm.append(y_test[i][j])
        temp_w10.append(well_10_test[i][j])
    linear_max_sm10.append(np.max(temp_sm))
    linear_min_sm10.append(np.min(temp_sm))
    linear_max_w10.append(np.max(temp_w10))
    linear_min_w10.append(np.min(temp_w10))

'''
Calculate R-squared and WMAPE for two producing wells
'''
# Load the pre-saved potential variables of the permeability coefficient field
ti_r2 = [219]
all_laten = []
for i in range(251):
    if i not in ti_r2:
        with open('../all_data/k_laten/var_0.45/no_{}'.format(i), 'rb') as f:
            laten = pickle.load(f)
            all_laten.append(torch.Tensor.detach(laten).cpu().numpy())

k_laten_test = np.array(all_laten[200:250], dtype='float64')[:,0,:]
well_9_test = np.array(well_9[200:250], dtype='float64')
well_10_test = np.array(well_10[200:250])

with open('../all_data/krg_model/var_0.45/KRG_wel9_model.pkl', 'rb') as f:
    sm = pickle.load(f)
y_test_9 = sm.predict_values(k_laten_test)
# Calculate R-squared of production well C1
R2_9 = []
for i in range(50):
    real = well_9_test[i]
    fake = y_test_9[i]
    fen_zi = 0
    fen_mu = 0
    for j in range(len(real)):  # 计算整个分子
        fen_zi_sig = np.power((real[j]-fake[j]), 2)
        fen_zi += fen_zi_sig
        fen_mu_sig = np.power((real[j]-np.mean(real)), 2)
        fen_mu += fen_mu_sig
    R = 1-(fen_zi/fen_mu)
    if R < 0.8:
        print(i,R)
    R2_9.append(R)
# Calculate WMAPE of production well C1
WMAPE_9 = []
for i in range(50):
    real = well_9_test[i]
    fake = y_test_9[i]
    fen_zi = 0
    fen_mu = 0
    for j in range(len(real)):
        fen_zi_sig = np.abs(real[j]-fake[j])
        fen_zi += fen_zi_sig
        fen_mu += real[j]
    WMAPE_9.append(fen_zi/fen_mu)
print('well9_r2 max:{} min:{} mean:{}'.format(np.max(R2_9),np.min(R2_9),np.mean(R2_9)))
print('well9_wmape max:{} min:{} mean:{}'.format(np.max(WMAPE_9),np.min(WMAPE_9),np.mean(WMAPE_9)))

# Calculate R-squared of production well C2
with open('../all_data/krg_model/var_0.45/KRG_wel10_model.pkl', 'rb') as f:
    sm = pickle.load(f)
y_test_10 = sm.predict_values(k_laten_test)

R2_10 = []
for i in range(50):
    real = well_10_test[i]
    fake = y_test_10[i]
    fen_zi = 0
    fen_mu = 0
    for j in range(len(real)):
        fen_zi_sig = np.power((real[j]-fake[j]), 2)
        fen_zi += fen_zi_sig
        fen_mu_sig = np.power((real[j]-np.mean(real)), 2)
        fen_mu += fen_mu_sig
    R = 1-(fen_zi/fen_mu)
    R2_10.append(R)

WMAPE_10 = []
for i in range(50):
    real = well_10_test[i]
    fake = y_test_10[i]
    fen_zi = 0
    fen_mu = 0
    for j in range(len(real)):
        fen_zi_sig = np.abs(real[j]-fake[j])
        fen_zi += fen_zi_sig
        fen_mu += real[j]
    WMAPE_10.append(fen_zi/fen_mu)
print('well10_r2 max:{} min:{} mean:{}'.format(np.max(R2_10),np.min(R2_10),np.mean(R2_10)))
print('well10_wmape max:{} min:{} mean:{}'.format(np.max(WMAPE_10),np.min(WMAPE_10),np.mean(WMAPE_10)))


# # start plotting
fig = plt.figure(figsize=(12, 9))
plt.rcParams['figure.dpi'] = 500
plt.rc('font',family='Times New Roman')
'''
C1 production
'''
# Subfigure 1
ax0 = fig.add_axes([.05, 0.72, .25, .23])
ax0.fill_between(np.arange(157), linear_max_sm9, linear_min_sm9, color='red',alpha=0.9, label='Phast')
ax0.fill_between(np.arange(157), linear_max_w9, linear_min_w9, color='blue',alpha=0.5, label='AAE-KRG')
ax0.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
ax0.legend(loc='upper right',fontsize=9)
ax0.set_xticks([0,25,50,75,100,125,150],[0,100,200,300,400,500,600])
ax0.set_yticks([0.,0.00005,0.0001,0.00015])
ax0.set_xlabel('days', fontsize=13, labelpad=0)
ax0.set_ylabel('U (mol/L)', fontsize=11,labelpad=0)
ax0.text(73,0.000147,'C 1', fontsize=12, color='black')
# # Subfigure 2
ax0 = fig.add_axes([.33, 0.72, .25, .23])
ax0.hist(R2_9, 50)
ax0.set_xlabel('R$^2$', fontsize=11, labelpad=0)
ax0.set_ylabel('Frequency', fontsize=11, labelpad=-4)
ax0.set_xticks([0.80,0.85,0.90,0.95,1.00])
ax0.set_yticks([0,2,4,6,8,10,12,14])
ax0.text(0.885,13.3,'C 1', fontsize=12, color='black')
# # Subfigure 3
ax1 = fig.add_axes([.61, 0.72, .25, .23])
ax1.hist(WMAPE_9, 50)
ax1.text(0.0605,7.63,'C 1', fontsize=12, color='black')
ax1.set_ylabel('Frequency', fontsize=11, labelpad=0)
ax1.set_xticks([0.00,0.02,0.04,0.06,0.08,0.10,0.12])
ax1.set_xlabel('WMAPE', fontsize=11)

'''
C2 production well
'''
# # Subfigure 4
ax0 = fig.add_axes([.05, 0.43, .25, .23])
ax0.fill_between(np.arange(157), linear_max_sm10, linear_min_sm10, color='red',alpha=0.9, label='Phast')
ax0.fill_between(np.arange(157), linear_max_w10, linear_min_w10, color='blue',alpha=0.5, label='AAE-KRG')
ax0.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
ax0.legend(loc='upper right',fontsize=9)
ax0.set_xticks([0,25,50,75,100,125,150],[0,100,200,300,400,500,600])
ax0.set_xlabel('days', fontsize=13, labelpad=0)
ax0.set_ylabel('U (mol/L)', fontsize=11,labelpad=0)
ax0.text(73,0.000395,'C 2', fontsize=12, color='black')
# ## Subfigure 5
ax0 = fig.add_axes([.33, 0.43, .25, .23])
ax0.hist(R2_10, 50)
ax0.set_xlabel('R$^2$', fontsize=11, labelpad=0)
ax0.set_ylabel('Frequency', fontsize=11, labelpad=-4)
ax0.text(0.938,11.45,'C 2', fontsize=12, color='black')
# Subfigure 6
ax1 = fig.add_axes([.61, 0.43, .25, .23])
ax1.hist(WMAPE_10, 50)
ax1.text(0.0545,7.62,'C 2', fontsize=12, color='black')
ax1.set_ylabel('Frequency', fontsize=11, labelpad=0)
ax1.set_xlabel('WMAPE', fontsize=11)

plt.savefig('./figuer_var0.45.png', dpi=500)
# plt.show()

