import pandas as pd
import numpy as np
from datetime import datetime
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
def deal_week(week):
    tmp=week.split()[0]
    res=''
    for i in range(len(tmp)):
        if(tmp[i]>='0' and tmp[i]<='9'):
            res+=tmp[i]
    return datetime.strptime(res, "%Y%m%d").weekday()
def deal_day(week):
    tmp=week.split()[0]
    res=''
    for i in range(len(tmp)):
        if(tmp[i]>='0' and tmp[i]<='9'):
            res+=tmp[i]
    return (datetime.strptime(res, "%Y%m%d").year-2018)*365+\
           (datetime.strptime(res, "%Y%m%d").month)*30+(datetime.strptime(res, "%Y%m%d").day)

def deal_beg(path):
    data = pd.read_csv(path,low_memory=False)
    data['page_tm_day'] = data['page_tm'].apply(lambda x: deal_day(x))
    first=np.min(data['page_tm_day'])
    log_max = pd.DataFrame(data.groupby(['id']).max()).reset_index()
    log_min = pd.DataFrame(data.groupby(['id']).min()).reset_index()
    log = pd.DataFrame(data.groupby(['id']).size()).reset_index()
    log.columns = ['id', 'beg_id_size']
    log_max['page_tm_last'] = log_max['page_tm_day'] - first
    log_max['page_tm_st'] = log_min['page_tm_day'] - first
    log_max['page_tm_max_min_beh'] = log_max['page_tm_last'] - log_max['page_tm_st']
    # log_max = log_max[['id', 'page_tm_max_min_beh', 'page_tm_last', 'page_tm_st']]
    log_max = log_max[['id', 'page_tm_max_min_beh']]
    log = log.merge(log_max, on=['id'], how='left')
    log['fre_beh'] = log['beg_id_size']/log['page_tm_max_min_beh']
    return log
#a=deal_beg('data/训练数据集_beh_con.csv')
def deal_trd(train_trd):
    train_trd.drop_duplicates(keep='first', inplace=True)
    # 每種支付方式花多少錢
    #
    train_trd['trx_tm_day']=train_trd['trx_tm'].apply(lambda x: deal_day(x))
    first=np.min(train_trd['trx_tm_day'])

    log_max = pd.DataFrame(train_trd.groupby(['id']).max()).reset_index()
    log_min = pd.DataFrame(train_trd.groupby(['id']).min()).reset_index()

    train_trd['trx_tm']=train_trd['trx_tm'].apply(lambda x:int(x.split()[1].split(':')[0]))
    log_time = pd.DataFrame(train_trd.groupby(['id', 'trx_tm']).size()).reset_index()
    log_time.columns = ['id', 'trx_tm', 'trx_tm_size']
    for i in list(set(log_time['trx_tm'])):
        name = "trx_tm_size_{}".format(i)
        tmp = pd.DataFrame(log_time[log_time['trx_tm'] == i])
        log_time[name] = tmp['trx_tm_size']
    log_time.fillna(0)
    log_time = log_time.groupby(['id']).sum()


    log_2= pd.DataFrame(train_trd.groupby(['id', 'Trx_Cod2_Cd']).size()).reset_index()
    log_2.columns = ['id', 'Trx_Cod2_Cd', 'Trx_Cod2_Cd_size']
    for i in list(set(log_2['Trx_Cod2_Cd'])):
        name = "Trx_Cod2_Cd_size_{}".format(i)
        tmp = pd.DataFrame(log_2[log_2['Trx_Cod2_Cd'] == i])
        log_2[name] = tmp['Trx_Cod2_Cd_size']
    log_2.fillna(0)
    log_2= log_2.groupby(['id']).sum()


    log_1= pd.DataFrame(train_trd.groupby(['id', 'Trx_Cod1_Cd']).size()).reset_index()
    log_1.columns = ['id', 'Trx_Cod1_Cd', 'Trx_Cod1_Cd_size']
    for i in list(set(log_1['Trx_Cod1_Cd'])):
        name = "Trx_Cod1_Cd_size_{}".format(i)
        tmp = pd.DataFrame(log_1[log_1['Trx_Cod1_Cd'] == i])
        log_1[name] = tmp['Trx_Cod1_Cd_size']
    log_1.fillna(0)
    log_1= log_1.groupby(['id']).sum()


    #支付方式
    log_3= pd.DataFrame(train_trd.groupby(['id', 'Dat_Flg3_Cd']).size()).reset_index()
    log_3.columns = ['id', 'Dat_Flg3_Cd', 'Dat_Flg3_Cd_size']
    for i in list(set(log_3['Dat_Flg3_Cd'])):
        name = "Dat_Flg3_Cd_size_{}".format(i)
        tmp = pd.DataFrame(log_3[log_3['Dat_Flg3_Cd'] == i])
        log_3[name] = tmp['Dat_Flg3_Cd_size']
    log_3.fillna(0)
    log_3 = log_3.groupby(['id']).sum()

    #交易方向
    log_sum = pd.DataFrame(train_trd.groupby(['id', 'Dat_Flg1_Cd']).sum()).reset_index()[
        ['id', 'Dat_Flg1_Cd', 'cny_trx_amt']]
    for i in list(set(log_sum['Dat_Flg1_Cd'])):
        name = "Dat_Flg1_Cd_sum_{}".format(i)
        tmp = pd.DataFrame(log_sum[log_sum['Dat_Flg1_Cd'] == i])
        log_sum[name] = tmp['cny_trx_amt']/10


    log = pd.DataFrame(train_trd.groupby(['id', 'Dat_Flg1_Cd']).mean()).reset_index()[
        ['id', 'Dat_Flg1_Cd', 'cny_trx_amt']]
    for i in list(set(log['Dat_Flg1_Cd'])):
        name = "Dat_Flg1_Cd_{}".format(i)
        tmp = pd.DataFrame(log[log['Dat_Flg1_Cd'] == i])
        log[name] = tmp['cny_trx_amt']

    log_4 = pd.DataFrame(train_trd.groupby(['id']).size()).reset_index()
    log_4.columns = ['id',  'id_size']

    del log['Dat_Flg1_Cd']
    del log['cny_trx_amt']
    del log_sum['Dat_Flg1_Cd']
    del log_sum['cny_trx_amt']
    #del log_var['cny_trx_amt']
    log.fillna(0)
    log_sum.fillna(0)
    log = log.groupby(['id']).mean()
    log_sum = log_sum.groupby(['id']).sum()

    log=log.merge(log_sum,on=['id'],how='left')
    log = log.merge(log_3, on=['id'], how='left')
    log = log.merge(log_1, on=['id'], how='left')
    log = log.merge(log_2, on=['id'], how='left')
    log = log.merge(log_time, on=['id'], how='left')
    log = log.merge(log_4, on=['id'], how='left')

    #log = log.merge(log_var, on=['id'], how='left')
    log['aa']=log['Dat_Flg1_Cd_sum_B']/(log['Dat_Flg1_Cd_sum_C']+1)

    log_max['trx_tm_last'] = log_max['trx_tm_day']-first
    log_max['trx_tm_st'] = log_min['trx_tm_day']-first

    log_max['trx_tm_max_min'] = log_max['trx_tm_last'] - log_max['trx_tm_st']

    log_max=log_max[['id','trx_tm_max_min','trx_tm_last']]
    #log_max=log_max[['id','trx_tm_max_min','trx_tm_last','trx_tm_st']]
    log = log.merge(log_max, on=['id'], how='left')

    log['fre']=log['trx_tm_max_min']/log['id_size']

    log['id_size'] = log['id_size'].apply(lambda x: np.log(x + 1))
    return log

def deal_tag(train_path,train_trd_path,test_path,test_trd_path,trd=True):
    train_tag = pd.read_csv(train_path)
    test_tag=pd.read_csv(test_path)
    label = train_tag.pop('flag')

    data_tag= pd.concat([train_tag, test_tag], axis=0)
    data_tag = data_tag.reset_index()
    if trd:
        print("添加trd")
        log_train = pd.read_csv(train_trd_path, low_memory=False)
        log_test = pd.read_csv(test_trd_path, low_memory=False)
        log_train=log_train.append(log_test,sort=False)
        log_train= deal_trd(log_train)
        data_tag = data_tag.merge(log_train, on=['id'], how='left')

    data_tag.replace("\\N", np.nan, inplace=True)
    data_tag['cur_debit_cnt'] = data_tag['cur_debit_cnt'].apply(lambda x: 20 if x >= 20 else x)  # 不好不壞
    data_tag['cur_credit_cnt'] = data_tag['cur_credit_cnt'].apply(lambda x: 20 if x >= 20 else x)  # 不好不壞
    data_tag['cur_debit_min_opn_dt_cnt'] = data_tag['cur_debit_min_opn_dt_cnt'] / 30  # 有點用
    data_tag['cur_credit_min_opn_dt_cnt'] = data_tag['cur_credit_min_opn_dt_cnt'] / 30  # 有點用
    data_tag['crd_card_act_ind'].fillna(0, inplace=True)  # 有用，有缺失值
    data_tag['crd_card_act_ind'] = data_tag['crd_card_act_ind'].astype('int')
    data_tag['l1y_crd_card_csm_amt_dlm_cd'].fillna(0, inplace=True)  #
    data_tag['l1y_crd_card_csm_amt_dlm_cd'] = data_tag['l1y_crd_card_csm_amt_dlm_cd'].astype('int')
    data_tag['atdd_type'].fillna(0, inplace=True)
    data_tag['atdd_type'] = data_tag['atdd_type'].astype('int')
    class_mapping = {'M': 0, 'F': 1, 'L': 2}
    data_tag['gdr_cd'] = data_tag['gdr_cd'].map(class_mapping)
    data_tag['mrg_situ_cd'] = data_tag['mrg_situ_cd'].fillna("L")
    class_mapping = {'B': 0, 'A': 1, 'Z': 2, 'O': 3, '~': 4, 'L': 5}
    data_tag['mrg_situ_cd'] = data_tag['mrg_situ_cd'].map(class_mapping)
    del data_tag['edu_deg_cd']
    data_tag['acdm_deg_cd'] = data_tag['acdm_deg_cd'].fillna("L")
    class_mapping = {'30': 0, 'Z': 1, '31': 2, 'G': 3, 'C': 4, 'F': 5, 'D': 6, 'L': 7}
    data_tag['acdm_deg_cd'] = data_tag['acdm_deg_cd'].map(class_mapping)
    del data_tag['deg_cd']
    data_tag['job_year'] = data_tag['job_year'].apply(lambda x: 0 if x == 99 else x)
    data_tag['cc'] = data_tag['cur_credit_cnt'] / (data_tag['cur_debit_cnt'] + 1)  # 有用
    data_tag['Dat_Flg1_Cd_C'].fillna(0, inplace=True)
    data_tag['Dat_Flg1_Cd_B'].fillna(0, inplace=True)

    train = data_tag[:train_tag.shape[0]]
    test = data_tag[train_tag.shape[0]:]

    train=train.reset_index()
    test=test.reset_index()
    return train,test,label
def deal_tag_old(path,trd_path=None,beh_path=None):

    train_tag = pd.read_csv(path,low_memory=False)
    train_tag.replace("\\N", np.nan, inplace=True)


    if trd_path:
        log_train = pd.read_csv(trd_path, low_memory=False)
        log=deal_trd(log_train)
        train_tag=train_tag.merge(log,on=['id'],how='left')
    # if beh_path:
    #     log = deal_beg(beh_path)
    #     train_tag = train_tag.merge(log, on=['id'], how='left')
    #1持有招行借记卡张数
    train_tag['cur_debit_cnt'] = train_tag['cur_debit_cnt'].apply(lambda x: 20 if x >= 20 else x)  # 不好不壞
    #2持有招行信用卡张数
    train_tag['cur_credit_cnt'] = train_tag['cur_credit_cnt'].apply(lambda x: 20 if x >= 20 else x)  # 不好不壞
    #3持有招行借记卡天数
    train_tag['cur_debit_min_opn_dt_cnt'] = train_tag['cur_debit_min_opn_dt_cnt'] / 30  # 有點用
    #4持有招行信用卡天数
    train_tag['cur_credit_min_opn_dt_cnt'] = train_tag['cur_credit_min_opn_dt_cnt'] /30  # 有點用
    #5招行借记卡持卡最高等级代码
    #train_tag['cur_debit_crd_lvl'] = train_tag['cur_debit_crd_lvl'] 不处理，没缺失值

    # 6招行信用卡持卡最高等级代码
    train_tag['hld_crd_card_grd_cd'] = train_tag['hld_crd_card_grd_cd'].fillna(0)#有缺失值
    train_tag['hld_crd_card_grd_cd'] = train_tag['hld_crd_card_grd_cd'].astype('int')

    #7信用卡活跃标识
    train_tag['crd_card_act_ind'].fillna(0, inplace=True)  # 有用，有缺失值
    train_tag['crd_card_act_ind'] = train_tag['crd_card_act_ind'].astype('int')
    #8最近一年信用卡消费金额分层
    train_tag['l1y_crd_card_csm_amt_dlm_cd'].fillna(0, inplace=True)#
    train_tag['l1y_crd_card_csm_amt_dlm_cd']=train_tag['l1y_crd_card_csm_amt_dlm_cd'].astype('int')

    #9信用卡还款方式
    train_tag['atdd_type'].fillna(0, inplace=True)#有用
    train_tag['atdd_type'] = train_tag['atdd_type'].astype('int')

    #10信用卡永久信用额度分层,没缺失值异常值
    #11年龄,不处理
    #12gdr_cd
    class_mapping = {'M': 0, 'F': 1, 'L': 2}
    train_tag['gdr_cd'] = train_tag['gdr_cd'].map(class_mapping)
    #13mrg_situ_cd
    train_tag['mrg_situ_cd'] = train_tag['mrg_situ_cd'].fillna("L")
    class_mapping = {'B': 0, 'A': 1, 'Z': 2, 'O': 3, '~': 4, 'L': 5}
    train_tag['mrg_situ_cd'] = train_tag['mrg_situ_cd'].map(class_mapping)

    #14edu_deg_cd
    del train_tag['edu_deg_cd']
    #15学历
    train_tag['acdm_deg_cd'] = train_tag['acdm_deg_cd'].fillna("L")
    class_mapping = {'30': 0, 'Z': 1, '31': 2, 'G': 3, 'C': 4, 'F': 5, 'D': 6, 'L': 7}
    train_tag['acdm_deg_cd'] = train_tag['acdm_deg_cd'].map(class_mapping)
    #16学位
    del train_tag['deg_cd']
    #17job_year
    train_tag['job_year'] = train_tag['job_year'].apply(lambda x: 0 if x == 99 else x)




    train_tag['cc'] = train_tag['cur_credit_cnt'] / (train_tag['cur_debit_cnt'] + 1)  # 有用
    train_tag['Dat_Flg1_Cd_C'].fillna(0,inplace=True)
    train_tag['Dat_Flg1_Cd_B'].fillna(0, inplace=True)



    return train_tag
#train_data = deal_tag_old(path="data/训练数据集_tag.csv",trd_path="data/训练数据集_trd.csv",beh_path="data/训练数据集_beh.csv")