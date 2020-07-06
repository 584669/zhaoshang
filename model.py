import lightgbm as lgb
import catboost as ctb
import numpy as np
import xgboost as xgb
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from catboost import Pool
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import BayesianRidge, HuberRegressor,LinearRegression,LogisticRegression
#from deepffm import DeepFM
def lgb_cv_rf(train_, test_, params,nfold,random_state=2019):
    train = train_.copy()
    test = test_.copy()
    y_train = train.pop('flag')
    X_train = train.values
    X_test = test.values
    oof_lgb = np.zeros(len(train))
    predictions_lgb = np.zeros(len(test))
    folds = KFold(n_splits=nfold, shuffle=True, random_state=random_state)
    splits = folds.split(X_train, y_train)

    for fold_, (trn_idx, val_idx) in enumerate(splits):
        print("fold n°{}".format(fold_ + 1))
        trn_data = lgb.Dataset(X_train[trn_idx], y_train[trn_idx])
        val_data = lgb.Dataset(X_train[val_idx], y_train[val_idx])
        num_round = 20000
        clf = lgb.train(params, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=100,
                        early_stopping_rounds=200)

        train_new_feature=lgb.Dataset(clf.predict(X_train[trn_idx], num_iteration=clf.best_iteration,pred_leaf=True),
                                      y_train[trn_idx])
        test_new_feature=clf.predict(X_test, num_iteration=clf.best_iteration,pred_leaf=True)
        val_new_feature=lgb.Dataset(clf.predict(X_train[val_idx], num_iteration=clf.best_iteration,pred_leaf=True),
                                    y_train[val_idx])

        model = lgb.train(params, train_new_feature, num_round, valid_sets=[train_new_feature, val_new_feature],
                          verbose_eval=100,early_stopping_rounds=200)


        oof_lgb[val_idx] = model.predict(val_new_feature,num_iteration=model.best_iteration)
        predictions_lgb += model.predict(test_new_feature,num_iteration=model.best_iteration) / folds.n_splits
        # oof_lgb[val_idx] = clf.predict(X_train[val_idx], num_iteration=clf.best_iteration)
        # predictions_lgb += clf.predict(X_test, num_iteration=clf.best_iteration) / folds.n_splits
        print("lgb auc score: {:<8.8f}".format(roc_auc_score(y_train[val_idx], oof_lgb[val_idx])))
    print("lgb auc score: {:<8.8f}".format(roc_auc_score(y_train, oof_lgb)))
    return oof_lgb,predictions_lgb

# def lgb_cv_ffm(train_, test_, params,nfold,random_state=2019):
#     dfm_params = {
#         "use_fm": True,
#         "use_deep": True,
#         "embedding_size": 8,
#         "dropout_fm": [1.0, 1.0],
#         "deep_layers": [32, 32],
#         "dropout_deep": [0.5, 0.5, 0.5],
#         "deep_layers_activation": tf.nn.relu,
#         "epoch": 30,
#         "batch_size": 1024,
#         "learning_rate": 0.001,
#         "optimizer_type": "adam",
#         "batch_norm": 1,
#         "batch_norm_decay": 0.995,
#         "l2_reg": 0.01,
#         "verbose": True,
#         "eval_metric": roc_auc_score,
#         "random_seed": 2017
#     }
#     train = train_.copy()
#     test = test_.copy()
#     y_train = train.pop('flag')
#     X_train = train.values
#     X_test = test.values
#     oof_lgb = np.zeros(len(train))
#     predictions_lgb = np.zeros(len(test))
#     folds = KFold(n_splits=nfold, shuffle=True, random_state=random_state)
#     splits = folds.split(X_train, y_train)
#
#     for fold_, (trn_idx, val_idx) in enumerate(splits):
#         print("fold n°{}".format(fold_ + 1))
#         trn_data = lgb.Dataset(X_train[trn_idx], y_train[trn_idx])
#         val_data = lgb.Dataset(X_train[val_idx], y_train[val_idx])
#         num_round = 20000
#         clf = lgb.train(params, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=100,
#                         early_stopping_rounds=200)
#
#         train_new_feature=clf.predict(X_train[trn_idx], num_iteration=clf.best_iteration,pred_leaf=True)
#         test_new_feature=clf.predict(X_test, num_iteration=clf.best_iteration,pred_leaf=True)
#         val_new_feature=clf.predict(X_train[val_idx], num_iteration=clf.best_iteration,pred_leaf=True)
#         print(np.array(train_new_feature).shape)
#         print(np.array(test_new_feature).shape)
#         dfm = DeepFM(**dfm_params)
#         enc = OneHotEncoder()
#         enc.fit(train_new_feature)
#
#         dfm.fit(train_new_feature, val_new_feature,y_train[trn_idx])
#
#         dfm.predict(Xi_valid, Xv_valid)
#
#         # evaluate a trained model
#         dfm.evaluate(Xi_valid, Xv_valid, y_valid)
#         lm = LogisticRegression(solver='lbfgs', max_iter=100)
#         lm.fit(enc.transform(train_new_feature), y_train[trn_idx])
#         oof_lgb[val_idx] = lm.predict(val_new_feature)
#         predictions_lgb += lm.predict(test_new_feature) / folds.n_splits
#         # oof_lgb[val_idx] = clf.predict(X_train[val_idx], num_iteration=clf.best_iteration)
#         # predictions_lgb += clf.predict(X_test, num_iteration=clf.best_iteration) / folds.n_splits
#         print("lgb auc score: {:<8.8f}".format(roc_auc_score(y_train[val_idx], oof_lgb[val_idx])))
#     print("lgb auc score: {:<8.8f}".format(roc_auc_score(y_train, oof_lgb)))
#     return oof_lgb,predictions_lgb
def stack(*avg):
    train_stack = np.vstack(avg[0]).transpose()
    test_stack = np.vstack(avg[1]).transpose()
    y_train=avg[2]
    folds_stack = StratifiedKFold(n_splits=10, shuffle=True, random_state=8888)
    oof_stack = np.zeros(train_stack.shape[0])
    predictions = np.zeros(test_stack.shape[0])
    for fold_, (trn_idx, val_idx) in enumerate(folds_stack.split(train_stack, y_train)):
        print("fold :", fold_ + 1)
        trn_data, trn_y = train_stack[trn_idx], y_train[trn_idx]
        val_data, val_y = train_stack[val_idx], y_train[val_idx]
        stacking = BayesianRidge()
        stacking.fit(trn_data, trn_y)
        oof_stack[val_idx] = stacking.predict(val_data)
        predictions += stacking.predict(test_stack) / folds_stack.n_splits

    print("stacking auc score: {:<8.8f}".format(roc_auc_score(y_train, oof_stack)))
    return predictions
def lgb_cv(train_, test_,y_train, params,nfold,random_state=2019):
    train = train_.copy()
    test = test_.copy()

    X_train = train.values
    X_test = test.values
    oof_lgb = np.zeros(len(train))
    predictions_lgb = np.zeros(len(test))
    folds = KFold(n_splits=nfold, shuffle=True, random_state=random_state)
    splits = folds.split(X_train, y_train)
    for fold_, (trn_idx, val_idx) in enumerate(splits):
        print("fold n°{}".format(fold_ + 1))
        trn_data = lgb.Dataset(X_train[trn_idx], y_train[trn_idx])
        val_data = lgb.Dataset(X_train[val_idx], y_train[val_idx])
        num_round = 20000
        clf = lgb.train(params, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=100,
                        early_stopping_rounds=200)
        oof_lgb[val_idx] = clf.predict(X_train[val_idx], num_iteration=clf.best_iteration)
        predictions_lgb += clf.predict(X_test, num_iteration=clf.best_iteration) / folds.n_splits

    print("lgb auc score: {:<8.8f}".format(roc_auc_score(y_train, oof_lgb)))
    return oof_lgb,predictions_lgb

def xgb_cv(train_, test_,  y_train ,params,nfold):
    train=train_.copy()
    test=test_.copy()

    X_train = train.values
    X_test = test.values
    oof_xgb = np.zeros(len(train))
    predictions_xgb = np.zeros(len(test))
    folds = KFold(n_splits=nfold, shuffle=True, random_state=2020)
    splits = folds.split(X_train, y_train)
    for fold_, (trn_idx, val_idx) in enumerate(splits):
        print("fold n°{}".format(fold_ + 1))
        trn_data = xgb.DMatrix(X_train[trn_idx], y_train[trn_idx])
        val_data = xgb.DMatrix(X_train[val_idx], y_train[val_idx])

        watchlist = [(trn_data, 'train'), (val_data, 'valid')]
        clf = xgb.train(dtrain=trn_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200,
                        verbose_eval=100, params=params)
        oof_xgb[val_idx] = clf.predict(xgb.DMatrix(X_train[val_idx]), ntree_limit=clf.best_ntree_limit)
        predictions_xgb += clf.predict(xgb.DMatrix(X_test), ntree_limit=clf.best_ntree_limit) / folds.n_splits
    print("xgb auc score: {:<8.8f}".format(roc_auc_score(y_train, oof_xgb)))
    return oof_xgb,predictions_xgb



def predicted(train,test,label,kind=['xgb']):
    y_train = label.values
    lgb_params= {
            'num_leaves': 64,
             'min_data_in_leaf': 50,
             'objective': 'binary',#'binary'
             'max_depth': 6,
             'learning_rate': 0.01,
             "boosting": "gbdt",#dart,gbdt,goss,rf
             "feature_fraction": 0.8,
             "bagging_freq": 1,
             "bagging_fraction": 0.8,
             "bagging_seed": 11,
             "metric": 'auc',
             "num_threads": 8,
             "verbosity": -1,
            "lambda_l1": 0.5,
         "lambda_l2": 5,
                 }
    lgb_params1 = {'num_leaves': 48,
                  'min_data_in_leaf': 50,
                  'objective': 'binary',  # 'binary'
                  'max_depth': 6,
                  'learning_rate': 0.01,
                  "boosting": "gbdt",
                  "feature_fraction": 0.8,
                  "bagging_freq": 1,
                  "bagging_fraction": 0.8,
                  "bagging_seed": 88,
                  "metric": 'auc',
                  "num_threads": 8,
                  "verbosity": -1,
                  "lambda_l1": 0.5,
                  "lambda_l2": 5,
                  }
    xgb_params = {
                  'eta': 0.01,
                  'max_depth': 6,
                  'subsample': 0.8,
                  'colsample_bytree': 0.8,
                  'objective': 'binary:logistic',
                  'eval_metric': 'auc',
                  'silent': True,
                  'nthread': 4,
                  'n_estimators': 20000,
                  'gamma': 0.1,
                  'min_child_weight': 25,
                  'num_threads': 8,
                  'alpha': 0.18,
                  'lambda': 0.23,
                  'colsample_bylevel': 0.8,
                  }


    train_stack = []
    test_stack = []
    if('xgb' in kind):
        oof_xgb, predictions_xgb = xgb_cv(train, test,y_train, xgb_params,5)
        train_stack.append(oof_xgb)
        test_stack.append(predictions_xgb)
    if('lgb' in kind):
        oof_lgb, predictions_lgb = lgb_cv(train, test,y_train, lgb_params, 10,random_state=2019)
        train_stack.append(oof_lgb)
        test_stack.append(predictions_lgb)
    if('lgb1' in kind):
        oof_lgb1, predictions_lgb1 = lgb_cv(train, test, y_train,lgb_params1, 5,random_state=888)
        train_stack.append(oof_lgb1)
        test_stack.append(predictions_lgb1)
    if ('lgb2' in kind):
        oof_lgb2, predictions_lgb2 = lgb_cv(train, test,y_train, lgb_params1, 10, random_state=8)
        train_stack.append(oof_lgb2)
        test_stack.append(predictions_lgb2)
    if(len(kind)>1):
        predictions = stack(train_stack, test_stack, y_train)
    else:
        predictions=test_stack[0]
    return predictions

def important(train_,kind=['xgb']):
    train = train_.copy()
    y_train = train.pop('flag')
    name=train.columns
    with open('import.txt', 'w',encoding='utf-8') as f:
        for i in range(1,len(name)+1):
            f.write("f{}:{}".format(i,name[i-1])+ '\n')
    X_train = train.values
    lgb_params = {'num_leaves': 48,
                  'min_data_in_leaf': 20,
                  'objective': 'binary',
                  'max_depth': -1,
                  'learning_rate': 0.01,
                  "boosting": "gbdt",
                  "feature_fraction": 0.9,
                  "bagging_freq": 1,
                  "bagging_fraction": 0.9,
                  "bagging_seed": 11,
                  "metric": 'auc',
                  "num_threads": 8,
                  "verbosity": -1}
    xgb_params = {'eta': 0.01,
                  'max_depth': 6,
                  'subsample': 0.8,
                  'colsample_bytree': 0.7,
                  'objective': 'reg:linear',
                  'eval_metric': 'rmse',
                  'silent': True,
                  'nthread': 4,
                  'n_estimators': 20000,
                  'gamma': 0.2,
                  'min_child_weight': 25,
                  'num_threads': 8,
                  'alpha': 0.18,
                  'lambda': 0.23,
                  'colsample_bylevel': 0.8,
                  }
    plt.figure(figsize=(20, 20))
    if('xgb' in kind):
        trn_data = xgb.DMatrix(X_train, y_train)
        model = xgb.train(dtrain=trn_data, num_boost_round=10000,params=xgb_params)
        xgb.plot_importance(model)
    elif('lgb' in kind):
        trn_data = lgb.Dataset(X_train, y_train)
        model = lgb.train(lgb_params, trn_data, 10000)
        lgb.plot_importance(model,figsize=(20,20))
    plt.savefig("importance_{}.png".format(kind[0]))