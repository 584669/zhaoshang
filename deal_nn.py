import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler
def deal_nn(train,test):
    train[['Trx_Cod2_Cd_size_129', 'Trx_Cod2_Cd_size_130', 'Trx_Cod2_Cd_size_131', 'Trx_Cod2_Cd_size_132',
           'Trx_Cod2_Cd_size_133', 'Trx_Cod2_Cd_size_134', 'Trx_Cod2_Cd_size_135', 'Trx_Cod2_Cd_size_136',
           'Trx_Cod2_Cd_size_301', 'Trx_Cod2_Cd_size_302', 'Trx_Cod2_Cd_size_303', 'Trx_Cod2_Cd_size_304',
           'Trx_Cod2_Cd_size_305', 'Trx_Cod2_Cd_size_306', 'Trx_Cod2_Cd_size_307', 'Trx_Cod2_Cd_size_308',
           'Trx_Cod2_Cd_size_309', 'Trx_Cod2_Cd_size_310', 'Trx_Cod2_Cd_size_311', 'Trx_Cod2_Cd_size_201',
           'Trx_Cod2_Cd_size_202', 'Trx_Cod2_Cd_size_203', 'Trx_Cod2_Cd_size_204', 'Trx_Cod2_Cd_size_205',
           'Trx_Cod2_Cd_size_206', 'Trx_Cod2_Cd_size_207', 'Trx_Cod2_Cd_size_208', 'Trx_Cod2_Cd_size_209',
           'Trx_Cod2_Cd_size_210', 'Trx_Cod2_Cd_size_211', 'Trx_Cod2_Cd_size_212', 'Trx_Cod2_Cd_size_213',
           'Trx_Cod2_Cd_size_101', 'Trx_Cod2_Cd_size_102', 'Trx_Cod2_Cd_size_103', 'Trx_Cod2_Cd_size_104',
           'Trx_Cod2_Cd_size_105', 'Trx_Cod2_Cd_size_106', 'Trx_Cod2_Cd_size_107', 'Trx_Cod2_Cd_size_108',
           'Trx_Cod2_Cd_size_109', 'Trx_Cod2_Cd_size_110', 'Trx_Cod2_Cd_size_111', 'Trx_Cod2_Cd_size_112',
           'Trx_Cod2_Cd_size_113', 'Trx_Cod2_Cd_size_114', 'Trx_Cod2_Cd_size_115', 'Trx_Cod2_Cd_size_116',
           'Trx_Cod2_Cd_size_117', 'Trx_Cod2_Cd_size_118', 'Trx_Cod2_Cd_size_122', 'Trx_Cod2_Cd_size_123',
           'Trx_Cod2_Cd_size_124', 'Trx_Cod2_Cd_size_125', 'Trx_Cod2_Cd_size_126', 'Trx_Cod2_Cd_size_127',
           'Dat_Flg1_Cd_sum_B', 'Dat_Flg1_Cd_sum_C', 'Dat_Flg3_Cd_size', 'Dat_Flg3_Cd_size_B',
           'Dat_Flg3_Cd_size_A', 'Dat_Flg3_Cd_size_C', 'Trx_Cod1_Cd', 'Trx_Cod1_Cd_size', 'Trx_Cod1_Cd_size_1',
           'Trx_Cod1_Cd_size_2', 'Trx_Cod1_Cd_size_3', 'Trx_Cod2_Cd', 'Trx_Cod2_Cd_size', 'Trx_Cod2_Cd_size_128',
           'trx_tm_size', 'trx_tm_size_0', 'trx_tm_size_1', 'trx_tm_size_2', 'trx_tm_size_3',
           'trx_tm_size_4', 'trx_tm_size_5', 'trx_tm_size_6', 'trx_tm_size_7', 'trx_tm_size_8',
           'trx_tm_size_9', 'trx_tm_size_10', 'trx_tm_size_11', 'trx_tm_size_12', 'trx_tm_size_13',
           'trx_tm_size_14', 'trx_tm_size_15', 'trx_tm_size_16', 'trx_tm_size_17', 'trx_tm_size_18',
           'trx_tm_size_19', 'trx_tm_size_20', 'trx_tm_size_21', 'trx_tm_size_22', 'trx_tm_size_23',
           'trx_tm', 'id_size', 'job_year', 'ic_ind', 'fr_or_sh_ind', 'l6mon_agn_ind', 'frs_agn_dt_cnt',
           'l12mon_buy_fin_mng_whl_tms', 'l12_mon_fnd_buy_whl_tms', 'loan_act_ind', 'ovd_30d_loan_tot_cnt',
           'hav_hou_grp_ind', 'vld_rsk_ases_ind', 'l12_mon_insu_buy_whl_tms', 'l12_mon_gld_buy_whl_tms',
           'his_lng_ovd_day', 'aa', 'trx_tm_max_min', 'trx_tm_last', 'fre'
           ]]=train[['Trx_Cod2_Cd_size_129', 'Trx_Cod2_Cd_size_130', 'Trx_Cod2_Cd_size_131', 'Trx_Cod2_Cd_size_132',
              'Trx_Cod2_Cd_size_133', 'Trx_Cod2_Cd_size_134', 'Trx_Cod2_Cd_size_135', 'Trx_Cod2_Cd_size_136',
              'Trx_Cod2_Cd_size_301', 'Trx_Cod2_Cd_size_302', 'Trx_Cod2_Cd_size_303', 'Trx_Cod2_Cd_size_304',
              'Trx_Cod2_Cd_size_305', 'Trx_Cod2_Cd_size_306', 'Trx_Cod2_Cd_size_307', 'Trx_Cod2_Cd_size_308',
              'Trx_Cod2_Cd_size_309', 'Trx_Cod2_Cd_size_310', 'Trx_Cod2_Cd_size_311', 'Trx_Cod2_Cd_size_201',
              'Trx_Cod2_Cd_size_202', 'Trx_Cod2_Cd_size_203', 'Trx_Cod2_Cd_size_204', 'Trx_Cod2_Cd_size_205',
              'Trx_Cod2_Cd_size_206', 'Trx_Cod2_Cd_size_207', 'Trx_Cod2_Cd_size_208', 'Trx_Cod2_Cd_size_209',
              'Trx_Cod2_Cd_size_210', 'Trx_Cod2_Cd_size_211', 'Trx_Cod2_Cd_size_212', 'Trx_Cod2_Cd_size_213',
              'Trx_Cod2_Cd_size_101', 'Trx_Cod2_Cd_size_102', 'Trx_Cod2_Cd_size_103', 'Trx_Cod2_Cd_size_104',
              'Trx_Cod2_Cd_size_105', 'Trx_Cod2_Cd_size_106', 'Trx_Cod2_Cd_size_107', 'Trx_Cod2_Cd_size_108',
              'Trx_Cod2_Cd_size_109', 'Trx_Cod2_Cd_size_110', 'Trx_Cod2_Cd_size_111', 'Trx_Cod2_Cd_size_112',
              'Trx_Cod2_Cd_size_113', 'Trx_Cod2_Cd_size_114', 'Trx_Cod2_Cd_size_115', 'Trx_Cod2_Cd_size_116',
              'Trx_Cod2_Cd_size_117', 'Trx_Cod2_Cd_size_118', 'Trx_Cod2_Cd_size_122', 'Trx_Cod2_Cd_size_123',
              'Trx_Cod2_Cd_size_124', 'Trx_Cod2_Cd_size_125', 'Trx_Cod2_Cd_size_126', 'Trx_Cod2_Cd_size_127',
           'Dat_Flg1_Cd_sum_B', 'Dat_Flg1_Cd_sum_C', 'Dat_Flg3_Cd_size', 'Dat_Flg3_Cd_size_B',
           'Dat_Flg3_Cd_size_A', 'Dat_Flg3_Cd_size_C', 'Trx_Cod1_Cd', 'Trx_Cod1_Cd_size', 'Trx_Cod1_Cd_size_1',
           'Trx_Cod1_Cd_size_2', 'Trx_Cod1_Cd_size_3', 'Trx_Cod2_Cd', 'Trx_Cod2_Cd_size', 'Trx_Cod2_Cd_size_128',
           'trx_tm_size', 'trx_tm_size_0', 'trx_tm_size_1', 'trx_tm_size_2', 'trx_tm_size_3',
           'trx_tm_size_4', 'trx_tm_size_5', 'trx_tm_size_6', 'trx_tm_size_7', 'trx_tm_size_8',
           'trx_tm_size_9', 'trx_tm_size_10', 'trx_tm_size_11', 'trx_tm_size_12', 'trx_tm_size_13',
           'trx_tm_size_14', 'trx_tm_size_15', 'trx_tm_size_16', 'trx_tm_size_17', 'trx_tm_size_18',
           'trx_tm_size_19', 'trx_tm_size_20', 'trx_tm_size_21', 'trx_tm_size_22', 'trx_tm_size_23',
           'trx_tm', 'id_size','job_year','ic_ind','fr_or_sh_ind','l6mon_agn_ind','frs_agn_dt_cnt',
           'l12mon_buy_fin_mng_whl_tms','l12_mon_fnd_buy_whl_tms','loan_act_ind', 'ovd_30d_loan_tot_cnt',
           'hav_hou_grp_ind','vld_rsk_ases_ind','l12_mon_insu_buy_whl_tms','l12_mon_gld_buy_whl_tms',
           'his_lng_ovd_day','aa','trx_tm_max_min','trx_tm_last', 'fre'
           ]].fillna(0)
    train[['gdr_cd','dnl_mbl_bnk_ind','hav_car_grp_ind']]=\
        train[['gdr_cd','dnl_mbl_bnk_ind','hav_car_grp_ind']].fillna(2)
    train[['dnl_bind_cmb_lif_ind','cust_inv_rsk_endu_lvl_cd' ]]=\
        train[['dnl_bind_cmb_lif_ind','cust_inv_rsk_endu_lvl_cd' ]].fillna(1)
    train[['fin_rsk_ases_grd_cd', 'confirm_rsk_ases_lvl_typ_cd', 'tot_ast_lvl_cd',
           'pot_ast_lvl_cd']]=train[['fin_rsk_ases_grd_cd','confirm_rsk_ases_lvl_typ_cd', 'tot_ast_lvl_cd',
           'pot_ast_lvl_cd' ]].fillna(-1)
    test[['Trx_Cod2_Cd_size_129', 'Trx_Cod2_Cd_size_130', 'Trx_Cod2_Cd_size_131', 'Trx_Cod2_Cd_size_132',
           'Trx_Cod2_Cd_size_133', 'Trx_Cod2_Cd_size_134', 'Trx_Cod2_Cd_size_135', 'Trx_Cod2_Cd_size_136',
           'Trx_Cod2_Cd_size_301', 'Trx_Cod2_Cd_size_302', 'Trx_Cod2_Cd_size_303', 'Trx_Cod2_Cd_size_304',
           'Trx_Cod2_Cd_size_305', 'Trx_Cod2_Cd_size_306', 'Trx_Cod2_Cd_size_307', 'Trx_Cod2_Cd_size_308',
           'Trx_Cod2_Cd_size_309', 'Trx_Cod2_Cd_size_310', 'Trx_Cod2_Cd_size_311', 'Trx_Cod2_Cd_size_201',
           'Trx_Cod2_Cd_size_202', 'Trx_Cod2_Cd_size_203', 'Trx_Cod2_Cd_size_204', 'Trx_Cod2_Cd_size_205',
           'Trx_Cod2_Cd_size_206', 'Trx_Cod2_Cd_size_207', 'Trx_Cod2_Cd_size_208', 'Trx_Cod2_Cd_size_209',
           'Trx_Cod2_Cd_size_210', 'Trx_Cod2_Cd_size_211', 'Trx_Cod2_Cd_size_212', 'Trx_Cod2_Cd_size_213',
           'Trx_Cod2_Cd_size_101', 'Trx_Cod2_Cd_size_102', 'Trx_Cod2_Cd_size_103', 'Trx_Cod2_Cd_size_104',
           'Trx_Cod2_Cd_size_105', 'Trx_Cod2_Cd_size_106', 'Trx_Cod2_Cd_size_107', 'Trx_Cod2_Cd_size_108',
           'Trx_Cod2_Cd_size_109', 'Trx_Cod2_Cd_size_110', 'Trx_Cod2_Cd_size_111', 'Trx_Cod2_Cd_size_112',
           'Trx_Cod2_Cd_size_113', 'Trx_Cod2_Cd_size_114', 'Trx_Cod2_Cd_size_115', 'Trx_Cod2_Cd_size_116',
           'Trx_Cod2_Cd_size_117', 'Trx_Cod2_Cd_size_118', 'Trx_Cod2_Cd_size_122', 'Trx_Cod2_Cd_size_123',
           'Trx_Cod2_Cd_size_124', 'Trx_Cod2_Cd_size_125', 'Trx_Cod2_Cd_size_126', 'Trx_Cod2_Cd_size_127',
           'Dat_Flg1_Cd_sum_B', 'Dat_Flg1_Cd_sum_C', 'Dat_Flg3_Cd_size', 'Dat_Flg3_Cd_size_B',
           'Dat_Flg3_Cd_size_A', 'Dat_Flg3_Cd_size_C', 'Trx_Cod1_Cd', 'Trx_Cod1_Cd_size', 'Trx_Cod1_Cd_size_1',
           'Trx_Cod1_Cd_size_2', 'Trx_Cod1_Cd_size_3', 'Trx_Cod2_Cd', 'Trx_Cod2_Cd_size', 'Trx_Cod2_Cd_size_128',
           'trx_tm_size', 'trx_tm_size_0', 'trx_tm_size_1', 'trx_tm_size_2', 'trx_tm_size_3',
           'trx_tm_size_4', 'trx_tm_size_5', 'trx_tm_size_6', 'trx_tm_size_7', 'trx_tm_size_8',
           'trx_tm_size_9', 'trx_tm_size_10', 'trx_tm_size_11', 'trx_tm_size_12', 'trx_tm_size_13',
           'trx_tm_size_14', 'trx_tm_size_15', 'trx_tm_size_16', 'trx_tm_size_17', 'trx_tm_size_18',
           'trx_tm_size_19', 'trx_tm_size_20', 'trx_tm_size_21', 'trx_tm_size_22', 'trx_tm_size_23',
           'trx_tm', 'id_size', 'job_year', 'ic_ind', 'fr_or_sh_ind', 'l6mon_agn_ind', 'frs_agn_dt_cnt',
           'l12mon_buy_fin_mng_whl_tms', 'l12_mon_fnd_buy_whl_tms', 'loan_act_ind', 'ovd_30d_loan_tot_cnt',
           'hav_hou_grp_ind', 'vld_rsk_ases_ind', 'l12_mon_insu_buy_whl_tms', 'l12_mon_gld_buy_whl_tms',
           'his_lng_ovd_day', 'aa', 'trx_tm_max_min', 'trx_tm_last', 'fre'
           ]]=test[['Trx_Cod2_Cd_size_129', 'Trx_Cod2_Cd_size_130', 'Trx_Cod2_Cd_size_131', 'Trx_Cod2_Cd_size_132',
           'Trx_Cod2_Cd_size_133', 'Trx_Cod2_Cd_size_134', 'Trx_Cod2_Cd_size_135', 'Trx_Cod2_Cd_size_136',
           'Trx_Cod2_Cd_size_301', 'Trx_Cod2_Cd_size_302', 'Trx_Cod2_Cd_size_303', 'Trx_Cod2_Cd_size_304',
           'Trx_Cod2_Cd_size_305', 'Trx_Cod2_Cd_size_306', 'Trx_Cod2_Cd_size_307', 'Trx_Cod2_Cd_size_308',
           'Trx_Cod2_Cd_size_309', 'Trx_Cod2_Cd_size_310', 'Trx_Cod2_Cd_size_311', 'Trx_Cod2_Cd_size_201',
           'Trx_Cod2_Cd_size_202', 'Trx_Cod2_Cd_size_203', 'Trx_Cod2_Cd_size_204', 'Trx_Cod2_Cd_size_205',
           'Trx_Cod2_Cd_size_206', 'Trx_Cod2_Cd_size_207', 'Trx_Cod2_Cd_size_208', 'Trx_Cod2_Cd_size_209',
           'Trx_Cod2_Cd_size_210', 'Trx_Cod2_Cd_size_211', 'Trx_Cod2_Cd_size_212', 'Trx_Cod2_Cd_size_213',
           'Trx_Cod2_Cd_size_101', 'Trx_Cod2_Cd_size_102', 'Trx_Cod2_Cd_size_103', 'Trx_Cod2_Cd_size_104',
           'Trx_Cod2_Cd_size_105', 'Trx_Cod2_Cd_size_106', 'Trx_Cod2_Cd_size_107', 'Trx_Cod2_Cd_size_108',
           'Trx_Cod2_Cd_size_109', 'Trx_Cod2_Cd_size_110', 'Trx_Cod2_Cd_size_111', 'Trx_Cod2_Cd_size_112',
           'Trx_Cod2_Cd_size_113', 'Trx_Cod2_Cd_size_114', 'Trx_Cod2_Cd_size_115', 'Trx_Cod2_Cd_size_116',
           'Trx_Cod2_Cd_size_117', 'Trx_Cod2_Cd_size_118', 'Trx_Cod2_Cd_size_122', 'Trx_Cod2_Cd_size_123',
           'Trx_Cod2_Cd_size_124', 'Trx_Cod2_Cd_size_125', 'Trx_Cod2_Cd_size_126', 'Trx_Cod2_Cd_size_127',
           'Dat_Flg1_Cd_sum_B', 'Dat_Flg1_Cd_sum_C', 'Dat_Flg3_Cd_size', 'Dat_Flg3_Cd_size_B',
           'Dat_Flg3_Cd_size_A', 'Dat_Flg3_Cd_size_C', 'Trx_Cod1_Cd', 'Trx_Cod1_Cd_size', 'Trx_Cod1_Cd_size_1',
           'Trx_Cod1_Cd_size_2', 'Trx_Cod1_Cd_size_3', 'Trx_Cod2_Cd', 'Trx_Cod2_Cd_size', 'Trx_Cod2_Cd_size_128',
           'trx_tm_size', 'trx_tm_size_0', 'trx_tm_size_1', 'trx_tm_size_2', 'trx_tm_size_3',
           'trx_tm_size_4', 'trx_tm_size_5', 'trx_tm_size_6', 'trx_tm_size_7', 'trx_tm_size_8',
           'trx_tm_size_9', 'trx_tm_size_10', 'trx_tm_size_11', 'trx_tm_size_12', 'trx_tm_size_13',
           'trx_tm_size_14', 'trx_tm_size_15', 'trx_tm_size_16', 'trx_tm_size_17', 'trx_tm_size_18',
           'trx_tm_size_19', 'trx_tm_size_20', 'trx_tm_size_21', 'trx_tm_size_22', 'trx_tm_size_23',
           'trx_tm', 'id_size', 'job_year', 'ic_ind', 'fr_or_sh_ind', 'l6mon_agn_ind', 'frs_agn_dt_cnt',
           'l12mon_buy_fin_mng_whl_tms', 'l12_mon_fnd_buy_whl_tms', 'loan_act_ind', 'ovd_30d_loan_tot_cnt',
           'hav_hou_grp_ind', 'vld_rsk_ases_ind', 'l12_mon_insu_buy_whl_tms', 'l12_mon_gld_buy_whl_tms',
           'his_lng_ovd_day', 'aa', 'trx_tm_max_min', 'trx_tm_last', 'fre'
           ]].fillna(0)
    test[['gdr_cd', 'dnl_mbl_bnk_ind', 'hav_car_grp_ind']]=\
        test[['gdr_cd', 'dnl_mbl_bnk_ind', 'hav_car_grp_ind']].fillna(2)
    test[['dnl_bind_cmb_lif_ind', 'cust_inv_rsk_endu_lvl_cd']]=\
        test[['dnl_bind_cmb_lif_ind', 'cust_inv_rsk_endu_lvl_cd']].fillna(1)
    test[['fin_rsk_ases_grd_cd', 'confirm_rsk_ases_lvl_typ_cd', 'tot_ast_lvl_cd',
          'pot_ast_lvl_cd']]=test[['fin_rsk_ases_grd_cd', 'confirm_rsk_ases_lvl_typ_cd', 'tot_ast_lvl_cd',
           'pot_ast_lvl_cd']].fillna(-1)
    # cols = [i for i in train.columns if i not in ['id', 'flag']]
    # for col in cols:
    #     if(train[col].isnull().sum()>0):
    #         print(col,train[col].isnull().sum())
    #     # print(train[col].describe())
    #     # print(train[col].value_counts())
    # feature=[i for i in test.columns if i not in ['id','flag']]
    # scaler = StandardScaler()
    # scaler.fit(train[feature].values)
    # train[feature] = scaler.transform(train[feature].values)
    # scaler = MinMaxScaler()
    # scaler.fit(test[feature].values)
    # test[feature] = scaler.transform(test[feature].values)
    # print(train.head(5))
    return train,test

