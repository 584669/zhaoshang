import pandas as pd
import lightgbm as lgb
import catboost as ctb
from model import lgb_cv, xgb_cv, stack, predicted, important
from data_mat import deal_tag, deal_tag_old
from deal_nn import deal_nn
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
if __name__ == '__main__':
    data_trd = pd.read_csv('data/训练数据集_trd.csv')

    trd = pd.read_csv('data/评分数据集_trd_b.csv')

    data_trd = data_trd.append(trd, sort=False)

    data_trd.to_csv('data/train_trd_con.csv', index=0)

    train_data = deal_tag_old(path="data/训练数据集_tag.csv", trd_path="data/train_trd_con.csv",
                              beh_path="data/训练数据集_beh.csv")
    test_data = deal_tag_old(path="data/评分数据集_tag_b.csv", trd_path="data/train_trd_con.csv",
                             beh_path="data/评分数据集_beh_b.csv")
    test_data.to_csv('data/test.csv', index=0)
    train_data.to_csv('data/train.csv', index=0)
    # train_data = pd.read_csv('data/train.csv')
    # test_data = pd.read_csv('data/test.csv')
    del train_data['id']
    result = pd.DataFrame()
    result['id'] = test_data.pop('id')
    y_train = train_data.pop('flag')
    result['score'] = predicted(train_data, test_data,y_train, kind=['lgb','lgb1'])  # ,'lgb','cat'
    result['score'] = result['score'].apply(lambda x: 1 if x >= 0.7 else round(x, 4))
    result['score'] = result['score'].apply(lambda x: 0 if x <= 0.05 else round(x, 4))
    with open('result.txt', 'w', encoding='utf-8') as f:
        for i in range(len(result)):
            f.write(str(result.loc[i, 'id']) + '\t' + str(result.loc[i, 'score']) + '\n')
