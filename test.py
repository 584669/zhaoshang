import pandas as pd
import lightgbm as lgb
import catboost as ctb
from model import xgb_cv, stack, predicted, important
from sklearn.metrics import mean_absolute_error,roc_auc_score
from sklearn.model_selection import KFold
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
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
        if(fold_==3):
            predictions_lgb += clf.predict(X_test, num_iteration=clf.best_iteration)
    print("lgb auc score: {:<8.8f}".format(roc_auc_score(y_train, oof_lgb)))
    return predictions_lgb
if __name__ == '__main__':
    lgb_params = {
        'num_leaves': 64,
        'min_data_in_leaf': 50,
        'objective': 'binary',  # 'binary'
        'max_depth': 6,
        'learning_rate': 0.01,
        "boosting": "gbdt",  # dart,gbdt,goss,rf
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
    train_data = pd.read_csv('data/train.csv')
    test_data = pd.read_csv('data/test.csv')
    del train_data['id']
    result = pd.DataFrame()
    result['id'] = test_data.pop('id')
    y_train = train_data.pop('flag')
    result['score'] = lgb_cv(train_data, test_data,y_train, lgb_params, 6,random_state=2019)
    result['score'] = result['score'].apply(lambda x: 1 if x >= 0.9 else round(x, 4))
    result['score'] = result['score'].apply(lambda x: 0 if x <= 0.1 else round(x, 4))
    with open('result.txt', 'w', encoding='utf-8') as f:
        for i in range(len(result)):
            f.write(str(result.loc[i, 'id']) + '\t' + str(result.loc[i, 'score']) + '\n')
