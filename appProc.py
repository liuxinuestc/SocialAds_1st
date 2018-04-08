#encoding:utf-8
__author__ = 'Xin'

import pandas as pd
import numpy as np
filePath = u'E:\\研究生\\SocialAds\\OriginalData\\'
fileSaveP = u'E:\\研究生\\SocialAds\\TencentData\\stageData\\'

def grpSize(filesIn, filesOut,grpby):
    """
    :param filesIn:进行groupby的文件是哪些，一个list
    :param filesOut:完成后保存在哪里
    :param grpby:按照什么特征进行grpby, 例如'appID'
    :return: 变量的分组统计次数
    """
    a = filesIn
    b = filesOut
    # user_installedapps.csv: userID,appID
    # user_app_actions.csv: userID,installTime,appID
    for i,j in enumerate(a):
        f = open(filePath+j)
        a = f.readline().strip().split(',')
        index = a.index(grpby)  #按照什么东西进行分组统计个数
        a = f.readline()
        appTimes = {}
        while(a):
            id = a.strip().split(',')[index]
            appTimes[id] = appTimes.get(id,0)+1
            a = f.readline()
        f.close()
        f = open(filePath+b[i]+'.csv','w')
        f.write(grpby+','+b[i]+'\n')
        for k in appTimes.iteritems():
            f.write(','.join(map(str,k))+'\n')
        f.close()

# a = ['user_installedapps.csv','user_app_actions.csv']  #数据文件
# b = ['appInstalledTimes','appActionTimes']
# grpSize(a,b,'appID')

def getAppCatDic():
    df = pd.read_csv(filePath+'app_categories.csv')
    appCat = dict(zip(df.appID,df.appCategory))
    return appCat

def getCreatXxDic(Xx):
    """
    :param Xx: creativeID对应的Xx
    :return:creativeID对应的Xx 字典，creativeID已经是数字型了
    """
    df = pd.read_csv(filePath+'ad.csv')
    creatXx = dict(zip(df['creativeID'],df[Xx]))
    return creatXx
# c = getAppCatDic()

def appClickTimes():
    f = open(filePath+'train.csv')
    creatAppDic = getCreatXxDic('appID')  #返回的字典key都是int型的
    appCatDic = getAppCatDic()
    f.readline()
    a = f.readline()
    label = []
    # clickTime = []
    creatID = []
    usrID = []
    appID =[]
    catID = []
    ctLimit = 260000
    dayRange = str(int(ctLimit/10000))
    # setApp = set([75, 160, 283, 304]) #有点击但是没有安装的app
    # 所有app，set([389, 262, 391, 137, 14, 271, 472, 146, 150, 25, 283, 284, 286, 160, 278, 419, 420, 421, 428, 304, 434, 442, 319, 195, 68, 198, 327, 328, 75, 205, 206, 336, 465, 83, 84, 88, 356, 218, 350, 293, 100, 229, 360, 105, 109, 113, 383, 116, 123, 127])
    #注意z中元素：0是label，1是点击时间，2是creativeID
    while(a):
        t = a.strip().split(',')
        ct = int(t[1])
        la = int(t[0])
        # if 1:
        # if setApp.__contains__(app):
        if ct < ctLimit:
            label.append(la)
            cr = int(t[3])
            app = creatAppDic[cr]
            # clickTime.append(ct)
            creatID.append(cr)
            usrID.append(int(t[4]))

            appID.append(app)
            catID.append(appCatDic[app])
        a = f.readline()
    f.close()

    # colm =['label','creativeID','appID','categoryID']
    # colm =['label','clickTime','creativeID','appID','categoryID']
    colm =['label','creativeID','appID','appCategory','userID']
    # table = pd.DataFrame(zip(label,creativeID,appID,categoryID),columns=colm)
    table = pd.DataFrame(list(zip(label,creatID,appID,catID,usrID)),columns=colm)

    # 得到数据中的用户cat安装点击次数
    cc = table.groupby(['userID','appCategory']).agg({'label': np.sum,'creativeID':'count'}).reset_index()
    cc['{0}ConvRate'.format('uc')] = cc['label']*1.0/cc['creativeID']
    cc.rename(columns={'label':'ucInstTimes','creativeID':'ucClickTimes'})\
        .to_csv(fileSaveP+'ucClickInstall{0}.csv'.format(dayRange),index=False)

    # 得到数据中的用户app安装点击次数
    cc = table.groupby(['appID','userID']).agg({'label': np.sum,'creativeID':'count'}).reset_index()
    cc['{0}ConvRate'.format('ua')] = cc['label']*1.0/cc['creativeID']
    cc.rename(columns={'label':'uaInstTimes','creativeID':'uaClickTimes'})\
        .to_csv(fileSaveP+'uaClickInstall{0}.csv'.format(dayRange),index=False)

    # table.groupby(['appID','userID','label']).size().reset_index().to_csv(filePath+'1111.csv',index=False)

    # 得到数据中的按类别安装点击次数
    for i in colm[2:]:
        cc = table.groupby(i).agg({'label':np.sum,'creativeID':'count'}).reset_index()
        cc['{0}ConvRate'.format(i[:4])] = cc['label']*1.0/cc['creativeID']
        cc.rename(columns={'label':'{0}ConvTimes'.format(i),'creativeID':'{0}ClickTimes'.format(i)}).\
            to_csv(fileSaveP+'{0}ClickConv{1}.csv'.format(i,dayRange),index=False)
    return
    # table = pd.DataFrame(zip(label,clickTime,creatID,appID,catID),columns=colm)
    # table = pd.DataFrame(zip(label,creatID,appID,catID,usrID),columns=colm)
    # print table[table['userID']==290299]
    for k in colm[2:]:
        grp = table.groupby(k)
        # 'ConvertRate.csv'
        f = open(filePath+k+'ConvertRate.csv','w')
        f.write(k+',convertRate\n')
        for i, j in grp:
            c = list(j['label'])
            rate = c.count('1')*1.0/len(c)
            f.write(str(i)+','+str(rate)+'\n')
        f.close()
    # table.groupby(['appID','userID']).size().reset_index().to_csv(filePath+'testGrpResult2.csv',index=False)
    return
    for i in colm[3:]:
        grpS = table.groupby([i,'label']).size().reset_index()
        grpS.columns=[i,'label','clickTimes']
        grpS[grpS['label']=='1']
        grpS.to_csv(filePath+i+'clickTimes.csv',index=False)

def appClickTimes2():
    """
    输出app广告的出现时间
    :return:
    """
    f = open(filePath+'train.csv')
    #注意z中元素：0是label，1是点击时间，2是creativeID
    creatAppDic = getCreatXxDic('appID')  #返回的字典key都是int型的
    appCatDic = getAppCatDic()
    f.readline()
    a = f.readline()
    app = {}

    count = 0
    while(a):
        t = a.strip().split(',')
        ct = int(t[1])

        if 1:
        # if ct < 270000:
            c = int(t[3])
            if not app.has_key(creatAppDic[c]):
                app[creatAppDic[c]] = ct
                count += 1
                if count>49:
                    s = list(app.items())
                    s = sorted(s,key=lambda t:t[1])
                    print (s)
                    return
        a = f.readline()
    f.close()

def appCatGrpSize():
    appCat = getAppCatDic()
    b = ['appInstalledTimes','appActionTimes']
    catInstall = {}
    for i in b:
        f = open(fileSaveP+i+'.csv')
        f.readline()
        a = f.readline()
        while(a):
            xx = map(int, a.strip().split(','))
            cat = appCat[xx[0]]
            catInstall[cat] = catInstall.get(cat,0) + xx[1]
            a = f.readline()
        f.close()
        f = open(filePath+i+'_cat.csv','w')
        f.write('appCategory,'+i.strip('app')+'\n')
        for k in catInstall.iteritems():
            f.write(','.join(map(str,k))+'\n')
        f.close()

def appActionTimes():
    tf = pd.read_csv(filePath+'user_app_actions.csv')
    z = tf.groupby('appID').size().reset_index()
    z.rename(columns={0:'actInstallTimes'}).to_csv(fileSaveP+'appIactionTimes31.csv',index=None)

    cc = tf[tf['installTime']>=280000]
    z = cc.groupby('appID').size().reset_index()
    z.rename(columns={0:'rec3actInstallTimes'}).to_csv(fileSaveP+'rec3appIactionTimes31.csv',index=None)

    aa = tf[tf['installTime']<260000]
    aa.groupby('appID').size().reset_index().rename(columns={0:'actInstallTimes'})\
        .to_csv(fileSaveP+'appIactionTimes26.csv',index=None)

    bb = aa[aa['installTime']>=230000]
    bb.groupby('appID').size().reset_index().rename(columns={0:'rec3actInstallTimes'})\
        .to_csv(fileSaveP+'rec3appIactionTimes26.csv',index=None)

# appCatGrpSize()

def fun(x):
    if x<0.01:
        return 0.0
    elif x<0.03:
        return 0.01
    elif x<0.2:
        return x
    else:
        return 0.99


# appClickTimes()
# appActionTimes()
a = pd.read_csv(filePath+'train.csv')[['creativeID','positionID']]
b = pd.read_csv(filePath+'train\\trainIDallfiltered4.csv')
b['creativeID'] = a['creativeID']
b['positionID']= a['positionID']
c = pd.read_csv(filePath+'ad.csv')[['creativeID','camgaignID','adID']]
b = b.merge(c,how = 'left',on='creativeID')
c = pd.read_csv(fileSaveP+'rec3appIactionTimes26.csv')
b = b.merge(c,how = 'left',on='appID')
c = pd.read_csv(fileSaveP+'appCategoryClickConv26.csv')
print('appCategoryClickConv26.csv')
b = b.merge(c,how = 'left',on='appCategory')
b.to_csv(filePath+'train\\trainIDallfiltered41.csv',index=None)

print('trainIDallfiltered41.csv\t'+str(b.columns))