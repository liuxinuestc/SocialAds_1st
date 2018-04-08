#encoding:utf-8
__author__ = 'Xin'

import pandas as pd


filePath = u'E:\\研究生\\SocialAds\\TencentData\\'


def usrAge():
    df = pd.read_csv(filePath+'user.csv')
    ageScales = [[0,18,22,30,40,50,81],
                 [0,15,18,22,25,30,35,40,50,81],  #[0,15,18,22,25,30,35,40,50,60,81],多一个就好了
                 [0,10,20,30,40,50,60,70,81]]
    global ageScale
    def func(x):
        global ageScale
        i = 0
        while(x>ageScale[i]):
            i += 1
        return i
    for i, ageScale in enumerate(ageScales):
        df['ageScale'+str(i)] = map(func, df['age'])
    del df['age']
    df.to_csv(filePath+'user1.csv',index=False)

# dealTocsv(filePath+'user1.csv')
# usrAge()
# advertiserID
def locateExtract():
    f = open(filePath+'user1.csv')
    f2 = open(filePath+'user2.csv','w')
    a = f.readline().strip()
    f2.write(a+',hometownProvince,residenceProvince\n')
    a = a.split(',')
    homeIndex = a.index('hometown')
    rIndex = a.index('residence')
    a = f.readline()

    while(a):
        f2.write(a.strip()+',')
        alist = a.split(',')
        home = alist[homeIndex]
        hP = home[:-2] if len(home)>2 else home
        r = alist[rIndex]
        rP = r[:-2] if len(r) > 2 else r
        f2.write(hP+','+rP+'\n')
        a = f.readline()
    f.close()
    f2.close()
# locateExtract()

def countUniqueXx(file,xx):
    f = open(file)
    a = f.readline().strip().split(',')
    ii = a.index(xx)
    b = set()
    a = f.readline().strip()
    while(a):
        b.add(a.split(',')[ii])
        a = f.readline().strip()
    f.close()
    print (len(b))

def avg_app_install_by_place(file,place):
    f = open(file)
    a = f.readline().strip().split(',')
    ii = a.index(place)
    iusr = a.index('userID')
    dicUserPlace = {}
    a = f.readline().strip()
    while(a):
        b = a.split(',')
        dicUserPlace[b[iusr]] = b[ii]
        a = f.readline().strip()
    f.close()
    dicPlaceCount = {}
    f = open(filePath+'usrFeat.csv')
    f.readline()
    a = f.readline().strip()
    while(a):
        b = a.split(',')
        c = sum(map(int,b[1:]))
        if dicPlaceCount.has_key(dicUserPlace[b[0]]):
            dicPlaceCount[dicUserPlace[b[0]]][0] += c
            dicPlaceCount[dicUserPlace[b[0]]][1] += 1
        else:
            dicPlaceCount[dicUserPlace[b[0]]] = [c,1]
        a = f.readline().strip()
    for i in dicPlaceCount.keys():
        dicPlaceCount[i][0] /= dicPlaceCount[i][1]*1.0
    f.close()
    f = open(filePath+'avgInstall%s.txt'%place,'w')
    f.write('{0},avgInstall{0}\n'.format(place))
    for i in dicPlaceCount.keys():
        f.write(i+','+str('%.2f'%dicPlaceCount[i][0])+'\n')
    f.close()

# f = filePath+'user2.csv'
# avg_app_install_by_place(f,'ageScale1')
#'hometown'  'residence'


# locateExtract()

# f = filePath+'user_installedapps.csv'
# countUniqueXx(f,'userID')
f = open(r'E:\研究生\SocialAds\OriginalData\train\testAlluca.csv')
instanceID = []
f.readline()
a = f.readline()
while(a):
    instanceID.append(int(a.split(',')[0]))
    a = f.readline()
f.close()
dd = pd.read_csv(filePath+'stageData\\result\\submission8\\submission1.csv')
print(dd.shape)
del dd['instanceID']
dd['instanceID'] = instanceID
tt = dd.groupby('instanceID').mean().reset_index()
print(tt.shape)
tt.to_csv(filePath+'stageData\\result\\submission8\\submission.csv',index=None)