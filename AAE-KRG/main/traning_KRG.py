
import pickle
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from smt.surrogate_models import KRG, XType
from smt.applications.mixed_integer import MixedIntegerKrigingModel
from smt.kriging import XSpecs

ens_no = 260  # 需要读取场的个数
start_num = 0
path_txt_9 = './data/well_data/well_9'    # leaching concentration over time of C1 well obtained from Phast
path_txt_10 = './data/well_data/well_10'    # C2
well_9 = []
well_10 = []
ti_chu = [42,80,99,126,155,173,176,192,237,240,]
for i in range(260):
    if i not in ti_chu:
        df_random_9 = pd.read_csv(path_txt_9+'/no_'+str(i+start_num)+'_9号井所有化学成分.txt', sep=' ')    # 提取后的均质场化学数据
        df_random_10 = pd.read_csv(path_txt_10+'/no_'+str(i+start_num)+'_10号井所有化学成分.txt', sep=' ')    # 提取后的均质场化学数据
        X = df_random_9['Time']
        # uranium concentration curve over tim
        wel_9_sig = np.array(df_random_9['U'])
        wel_10_sig = np.array(df_random_10['U'])
        well_9.append(wel_9_sig)
        well_10.append(wel_10_sig)

all_laten = []
for i in range(250):    # load latent variable
    with open('./data/k_laten/no_{}'.format(i), 'rb') as f:
        laten = pickle.load(f)
        all_laten.append(torch.Tensor.detach(laten).cpu().numpy())

# start build KRG model for two production wells
k_laten_train = np.array(all_laten[0:200], dtype='float64')[:,0,:]
k_laten_test = np.array(all_laten[200:250], dtype='float64')[:,0,:]
well_9_train = np.array(well_9[0:200], dtype='float64')
well_9_test = np.array(well_9[200:250], dtype='float64')
well_10_train = np.array(well_10[0:200])
well_10_test = np.array(well_10[200:250])
# KRG of C1 production well
from smt.surrogate_models import KRG
sm = KRG(theta0=[1e-2], poly='linear')
sm.set_training_values(k_laten_train, well_9_train)
sm.train()
y_test = sm.predict_values(k_laten_test)
# save model
with open('./data/KRG_model/KRG_wel9_model.pkl', 'wb') as f:
    pickle.dump(sm, f)

# KRG of C2
from smt.surrogate_models import KRG
sm = KRG(theta0=[1e-2], poly='linear')
sm.set_training_values(k_laten_train, well_10_train)
sm.train()
y_test = sm.predict_values(k_laten_test)
# save model
with open('./data/KRG_model/KRG_wel10_model.pkl', 'wb') as f:
    pickle.dump(sm, f)