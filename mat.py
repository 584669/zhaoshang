import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data_trd = pd.read_csv('data/训练数据集_trd.csv')


trd=pd.read_csv('data/评分数据集_trd_b.csv')



data_trd=data_trd.append(trd,sort=False)


data_trd.to_csv('data/train_trd_con.csv',index=0)

# data_trd = pd.read_csv('data/训练数据集_trd.csv')
# test_trd=pd.read_csv('data/评分数据集_trd.csv')
# data_trd=data_trd.append(test_trd,sort=False)
# data_trd.to_csv('data/训练数据集_trd_con.csv',index=0)
