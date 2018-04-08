#encoding:utf-8
import pandas as pd
import xgboost as xgb
import scipy as sp
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

filePath = u'E:\\研究生\\SocialAds\\TencentData\\stageData\\'
forigin = u'E:\\研究生\\SocialAds\\OriginalData\\'

def pathWalk(rootDir):
    """
    本函数得到rootDir下的所有文件的绝对路径
    :param rootDir:目录形式的str
    :return:目录下的所有的文件绝对路径
    """
    list_dirs = os.walk(rootDir)
    filelist = []
    for root, dirs, files in list_dirs:
        for f in files:
            filelist.append(os.path.join(root, f))
    return filelist

def sample2accordRate(datalabel,nfold):
    """
    本代码实现将数据按照类别分层抽样成nfold个不交叉的样本
    :param datalabel:数据的标签
    :param nfold:需要得到的数据样本个数
    :return:数据抽样的分配，返回的是数据索引
    """
    inlabel0 = 0
    inlabel1 = 0
    indexes = [[] for i in range(nfold)]
    for i,j in enumerate(datalabel):
        if j==0:
            indexes[inlabel0].append(i)
            inlabel0 = (inlabel0+1)%nfold
        else:
            indexes[inlabel1].append(i)
            inlabel1 = (inlabel1+1)%nfold
    return indexes

def logloss(pred, dtrain):
      act = dtrain.get_label()
      epsilon = 1e-15
      pred = sp.maximum(epsilon, pred)
      pred = sp.minimum(1-epsilon, pred)
      ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
      ll = ll * -1.0/len(act)
      return 'logloss',ll

def loadData(fname):
    """
    新的加入26数据作为模型补充数据，27以后作为需要预测的数据
    :param fname:
    :return:
    """
    # 专门用于测试的文件 trainIDall0head.csv
    df = pd.read_csv(forigin+'train\\train'+fname)
    # df = df.fillna(df.median()[['userIDavgClick2Conv26','appIDavgClick2Conv26']])
    # df = df.fillna(0)
    delist = ['conversionTime']
    for i in delist:
        if i in list(df.columns):
            del df[i]
    # for c in df.columns[1:]:
    #     maxd = max(df[c])
    #     mind = min(df[c])
    #     z = maxd - mind
    #     if z > 0:
    #         df[c] = (df[c] - mind)*1.0/ z
    #     else:
    #         df[c] = np.zeros(len(df))
    train = df[df['clickTime']<270000]  #0.66 被norm之后才这样
    ddtest = df[df['clickTime']>=270000]
    del train['clickTime']
    del ddtest['clickTime']

    Y = list(train['label'])
    del train['label']
    y_test = list(ddtest['label'])
    del ddtest['label']
    X_train, X_eval,y_train,y_eval = train_test_split(train.get_values(),Y,train_size=0.8)

    return X_train, X_eval,ddtest.get_values(),y_train,y_eval,y_test

def aa2Load(fname):
    """
    原始的数据划分形式
    :return:时间上毫不交叉的数据集
    """
    # "trainIDall0.csv"  "train.csv"
    data = pd.read_csv(forigin+"train\\train"+fname)  #trainStage里面只有270000以前的数据，没有以后的
    # ,'ucClickTimes26','ucConvRate26','ucInstTimes26'
    # ,'advertiserID','appID','camgaignID','adID','positionID','userID','creativeID'
    # data['rec3actInstallTimes'] =  data['rec3actInstallTimes'].fillna(0)
    # data['appIDClickTimes'] = data['appIDClickTimes'].fillna(0)
    # data = data.fillna(data.median()[['userIDavgClick2Conv26','appIDavgClick2Conv26']])
    data.fillna({'recInsCount7':0,'recInsCount21':0},inplace=True)
    data['sum'].fillna(data['sum'].mean(),inplace=True)

    print(data.shape)
    del data['userID']
    # del data['max']
    del data['appPlatform']
    # del data['creativeID']
    # del data['appCategoryConvTimes']
    # dellist = ['conversionTime', 'creativeID', 'userID','uaConvRate',
    #          'appCategoryConvTimes', 'appCategoryClickTimes','appIDClickTimes',
    #          'uaInstTimes','ucInstTimes','ucClickTimes','marriageStatus','haveBaby',
    #          'ageScale2', 'hometownProvince', 'residenceProvince','ageScalerank',
    #          'userIDavgClick2Conv', 'userIDConvTimes','recentInsTime','recentCategory',#'recInsCount7',
    #          'recInsCount14','recInsCountAll', 'clickTimes',
    #          'trans_time', 'rate']
    # for ii in dellist:
    #     if ii in list(data.columns):
    #         del data[ii]

    train = pd.DataFrame()
    ddtest = pd.DataFrame()
    ddeval = pd.DataFrame()

    if 'Norm' in fname:
        # norm版本
        train = data[data['clickTime']<0.6913]
        ddeval = data[(data['clickTime']>0.6913) & (data['clickTime']<0.7667)]
        ddtest = data[data['clickTime']>0.7667]
    else:
        # 非norm版本
        # train = data[data['clickTime']<261500]
        # ddeval = data[(data['clickTime']>=261500) & (data['clickTime']<269000)]
        # ddtest = data[data['clickTime']>269000]

        train = data[(data['clickTime']>=259000) &(data['clickTime']<279000)] #(data['clickTime']>=210000) &
        ddeval = data[(data['clickTime']>279000) & (data['clickTime']<289000)]
        ddtest = data[(data['clickTime']>269000)] #& (data['clickTime']<289000)

    del train['clickTime']
    del ddeval['clickTime']
    del ddtest['clickTime']
    data =None

    label = train['label']
    del train['label']

    evalLabel = ddeval['label']
    del ddeval['label']

    testLabel = ddtest['label']
    del ddtest['label']
    return train,ddeval,ddtest,label,evalLabel,testLabel

def aa2(fname):
    X_train,X_eval,X_test,y_train,y_eval,y_test = aa2Load(fname) #加入部分26数据训练,函数里面默认train路径
    # X_train,X_eval,X_test,y_train,y_eval,y_test = loadData(fname)   #随机加入26训练

    ddMat = xgb.DMatrix(X_train,y_train)
    ddevalMat = xgb.DMatrix(X_eval,y_eval)
    ddtestMat = xgb.DMatrix(X_test,y_test)

    print(ddtestMat.num_row())
    print(u'开始')

    params = {}
    params["booster"] = "gbtree"
    # params["objective"] = "reg:linear"  # 线性回归
    params["objective"] = "reg:logistic"  # 逻辑回归
    params["eta"] = 0.15  # 0.1
    params["gamma"] = 0.0001
    params["max_depth"] = 7  # 默认为6
    params['scale_pos_weight'] = 1
    params['colsample_bytree'] = 0.5
    params['subsample'] = 0.7
    params["silent"] = 1
    params['alpha']=1
    params['eval_metric ']=['rmse']
    num_round=50
    # watchlist= [(ddMat.slice(indexTrain), 'train'),(ddMat.slice(indexTest), 'val')]
    # ttt = ddMat.slice(sampleIndex[3])
    watchlist= [(ddMat, 'train'),(ddevalMat, 'val')]
    plst = list(params.items())  # Using 5000 rows for early stopping.
    print("训练开始\n")
    lr = LinearRegression(normalize=True)
    X_train = X_train.fillna(0)
    lr.fit(X_train,y_train)
    X_test = X_test.fillna(0)
    rlr = lr.predict(X_test)
    rf = RandomForestRegressor(11,max_depth=9,min_impurity_split=0.000001,oob_score=True)
    rf.fit(X_train,y_train)
    rrf = rf.predict(X_test)
    print(type(rrf))
    bst = xgb.train(plst,ddMat,num_boost_round=num_round,evals=watchlist,early_stopping_rounds=20,feval=logloss)

    print("\n预测开始\n")
    rbst = bst.predict(ddtestMat,ntree_limit=bst.best_ntree_limit)  #,ntree_limit=bst.best_ntree_limit
    result = list(map(np.mean,list(zip(rlr,rrf,rbst))))
    print(len(result))

    print(logloss(result,ddtestMat))
    return

    # test = pd.read_csv(forigin+"train\\test"+fname)
    # delist = ['instanceID','clickTime','label','userID','creativeID','ucConvRate'
    # ,'clickTimes_bf31','trans_time31','rate_bf31']#,'ucInstTimes','ucConvRate','uaConvRate','userIDavgClick2Conv'],'ucConvRate','uaConvRate','userIDavgClick2Conv'

    # for ii in delist:
    #     if ii in list(test.columns):
    #         del test[ii]
    # # dtest = xgb.DMatrix(test.get_values())
    #
    # test.fillna({'recInsCount7':0,'recInsCount21':0},inplace=True)
    # test['sum'].fillna(test['sum'].mean(),inplace=True)
    # dtest = xgb.DMatrix(test)
    #
    # # 结果预测
    # print("\n预测开始\n")
    # result = bst.predict(dtest)
    # print(result)
    #
    # f = open(forigin+'mergeLog.log','a')
    # f.write('\nsubmission4\t'+str(test.columns)+'\n')
    # f.close()
    #
    # if not os.path.exists(filePath+'result\\submission8\\'):
    #     os.makedirs(filePath+'result\\submission8\\')
    # f = open(filePath+"result\\submission8\\submission1.csv",'w')
    # f.write('instanceID,prob\n')
    # for i,j in enumerate(result):
    #     f.write(str(i+1)+","+str(j)+'\n')
    # f.close()
    # f.close()
    # print(len(result))
    # bst.dump_model(filePath+'xxx7.txt')

    z = list(sorted(bst.get_fscore().items(),key= lambda t:t[1]))
    f = open(filePath+'attrFscore9.txt','w')
    for i in z:
        f.write(i[0]+','+str(i[1])+'\n')
    f.close()
    return

def mergeData(stage):
    # train = pd.read_csv(forigin+'trainIdOnly.csv')
    fname = 'testIDall'
    train = pd.read_csv(forigin+fname+'.csv')

    # 原始测试
    # ad = pd.read_csv(forigin+'ad.csv')
    # appCat = pd.read_csv(forigin+'app_categories.csv')
    # train = train.merge(ad,how='left',on='creativeID')
    # train = train.merge(appCat,how='left',on='appID')
    # train.to_csv(forigin+'trainIDall.csv',index=None)

    # merged = pd.read_csv(filePath+'ucClickInstall26.csv')  #'appCategory','userID'
    merged = pd.read_csv(filePath+'appIDavgClick2Conv31.csv')  #'appID'
    # merged = pd.read_csv(filePath+'userIDavgClick2Conv26.csv')  #'userID'

    train = train.merge(merged,how='left',on=['appID'])
    train.to_csv(forigin+'train\\{0}{1}.csv'.format(fname,stage),index=None)
    f = open(forigin+'mergeLog.log','a')
    f.write('\n'+stage+'test\t'+str(train.columns)+'\n')
    f.close()

def mergeStrategy(fname):
    if 'appC' in fname :
        return 'appCategory'
    elif 'app' in fname:
        return 'appID'
    elif 'ua' in fname:
        return ['userID','appID']
    elif 'uc' in fname:
        return ['userID','appCategory']
    elif 'user' in fname:
        return 'userID'
    else:
        return None

def mergeAlluca(fname):
    flist = pathWalk(filePath)
    base = pd.read_csv(forigin+fname+'IDall.csv')
    # base = base[base['clickTime']<270000] 这个貌似没什么意义
    for i in flist:
        if ('26' in i) or ('31' not in i):  #train的条件
        # if ('26' not in i) or ('31' in i):  #test的条件
            mergeCondition = mergeStrategy(i)
            if mergeCondition!=None:
                print('merge {}\n'.format(i))
                merged = pd.read_csv(i)
                base = base.merge(merged,how='left',on=mergeCondition)
    base.to_csv(forigin+'train\\{}Alluca.csv'.format(fname),index=None)
    f = open(forigin+'mergeLog.log','a')
    f.write('\n{}Alluca.csv\t'.format(fname)+str(base.columns)+'\n')
    f.close()

def mergeUser():
    base = pd.read_csv(filePath+'user2.csv')
    for i in 'hometown,residence,ageScale1'.split(','):
        merged = pd.read_csv(filePath+'avgInstall{}.csv'.format(i))
        base= base.merge(merged,how='left',on=i)
    base.to_csv(filePath+'user3.csv',index=None)

if __name__ == '__main__':

    # mergeAlluca('train')
    # mergeData('2')
    # for i in 'f_reg,pearson,randLasso'.split(',')[2:]:
    #     aa2('trainFilleducaNorm20{0}best.csv'.format(i))
    aa2('IDallfiltered41.csv')
    # aa2('trainIDall1.csv')

