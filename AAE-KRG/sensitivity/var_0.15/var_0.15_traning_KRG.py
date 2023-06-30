
import pickle
import torch
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

ens_no = 260
start_num = 0
path_txt_9 = '../all_data/well_data/var_0.15/well_9'    # 提取文件后保存为txt文件的位置
path_txt_10 = '../all_data/well_data/var_0.15/well_10'
well_9 = []
well_10 = []
ti_chu = [42,80,99,126,155,173,176,192,237,240,]
for i in range(260):
    if i not in ti_chu:
        df_random_9 = pd.read_csv(path_txt_9+'/no_'+str(i+start_num)+'_9号井所有化学成分.txt', sep=' ')    # 提取后的均质场化学数据
        df_random_10 = pd.read_csv(path_txt_10+'/no_'+str(i+start_num)+'_10号井所有化学成分.txt', sep=' ')    # 提取后的均质场化学数据
        X = df_random_9['Time']
        wel_9_sig = np.array(df_random_9['U'])
        wel_10_sig = np.array(df_random_10['U'])
        well_9.append(wel_9_sig)
        well_10.append(wel_10_sig)

all_laten = []
for i in range(260):
    if i not in ti_chu:
        with open('../all_data/k_laten/var_0.15/no_{}'.format(i), 'rb') as f:
            laten = pickle.load(f)
            all_laten.append(torch.Tensor.detach(laten).cpu().numpy())
# Start building the kriging model
k_laten_train = np.array(all_laten[0:200], dtype='float64')[:,0,:]
k_laten_test = np.array(all_laten[200:250], dtype='float64')[:,0,:]
well_9_train = np.array(well_9[0:200], dtype='float64')
well_9_test = np.array(well_9[200:250], dtype='float64')
well_10_train = np.array(well_10[0:200])
well_10_test = np.array(well_10[200:250])

# KRG of C1 production well
from smt.surrogate_models import KRG
sm_wel9 = KRG(theta0=[1e-2], poly='linear')
sm_wel9.set_training_values(k_laten_train, well_9_train)
sm_wel9.train()
y_test = sm_wel9.predict_values(k_laten_test)
with open('../all_data/krg_model/var_0.15/KRG_wel9_model.pkl', 'wb') as f:
    pickle.dump(sm_wel9, f)

# KRG of C2
from smt.surrogate_models import KRG
sm_wel10 = KRG(theta0=[1e-2], poly='linear')
sm_wel10.set_training_values(k_laten_train, well_10_train)
sm_wel10.train()
y_test = sm_wel10.predict_values(k_laten_test)
with open('../all_data/krg_model/var_0.15/KRG_wel10_model.pkl', 'wb') as f:
    pickle.dump(sm_wel10, f)