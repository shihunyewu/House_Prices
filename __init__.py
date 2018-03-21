import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import *
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.ensemble import  *
from sklearn.model_selection import *

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
print(test_data.shape)
print(train_data.shape)
# train_data.info()
# print(train_data.describe())

'''
corrmat = df.corr()
f, ax = plt.subplots(figsize = (10, 10))
# 绘制热力图
sns.heatmap(corrmat, vmax=.7, square=True)
plt.show()
'''
# 通过上面的热力图可以发现有几个特征有很高的相关性
# 我们需要去除那些相关性很强的特征
# 首先去除目标数据
target_train = train_data['SalePrice']
train_data = train_data.drop('SalePrice',axis=1)  # axis=1表示列

threshold = 0.7
dataset_orginal = train_data
col_corr = set()
corr_matrix = train_data.corr()

for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if corr_matrix.iloc[i,j] >= threshold: # 如果相关性很高
            colname = corr_matrix.columns[i]   # 得到列名
            col_corr.add(colname)               # 将结果添加到col_corr中？
            if colname in train_data.columns:   # 如果 colname 在train_data 中
                del train_data[colname]         # 从dataset中删除 colname
test_data = test_data[train_data.columns.values.tolist()]
print(test_data.shape)
print(train_data.shape)
# print(col_corr)



'''
(1460, 81)
(1460, 76)
{'1stFlrSF', 'GarageYrBlt', 'TotRmsAbvGrd', 'GarageArea'}
'''

# 重新画热力图，发现相关性强的属性被删除
"""
corrmat = train_data.corr()
f, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(corrmat, vmax=.7, square=True);
plt.show()
"""

# 去除了相关性强的属性之后，我们现在可观察一下空值
# 如果一个属性行有很多的控制，那么我们可以将这一属性删除

# type(train_data.isnull().sum()) 显示为 Series
total = train_data.isnull().sum().sort_values(ascending=False)
percent = (100*(train_data.isnull().sum()/train_data.isnull().count())).sort_values(ascending=False)
# 将数据在 axis = 1 上重新连接，列名分别为 Total 和 Percent
missing_data = pd.concat([total,percent],axis= 1,keys=['Total','Percent'])
# 观察前 30 个情况
# print(missing_data.head(30))

print(test_data.shape)
print(train_data.shape)
# 观察可以发现，PoolQC这个属性数据缺失率高达 99.52%
"""
               Total    Percent
PoolQC          1453  99.520548
MiscFeature     1406  96.301370
Alley           1369  93.767123
Fence           1179  80.753425
FireplaceQu      690  47.260274
LotFrontage      259  17.739726
GarageCond        81   5.547945
GarageType        81   5.547945
GarageFinish      81   5.547945
GarageQual        81   5.547945
BsmtExposure      38   2.602740
BsmtFinType2      38   2.602740
BsmtFinType1      37   2.534247
BsmtQual          37   2.534247
BsmtCond          37   2.534247
MasVnrArea         8   0.547945
MasVnrType         8   0.547945
Electrical         1   0.068493
Exterior2nd        0   0.000000
YearBuilt          0   0.000000
ExterQual          0   0.000000
ExterCond          0   0.000000
Foundation         0   0.000000
Exterior1st        0   0.000000
RoofMatl           0   0.000000
RoofStyle          0   0.000000
YearRemodAdd       0   0.000000
SaleCondition      0   0.000000
OverallCond        0   0.000000
OverallQual        0   0.000000
"""

# 将缺失率超过 20% 的空数据删除
train_data = train_data.drop((missing_data[missing_data['Percent']>20]).index,1)
# 查看缺失数据最多的
test_data = test_data[train_data.columns.values.tolist()]
print(train_data.isnull().sum().max())
'''
259
'''
print('将缺失率超过 20% 的空数据删除')
print(test_data.shape)
print(train_data.shape)
# 将分散的类别信息转换成数据信息，也就是 one-hot编码

h = test_data.shape

total_data = pd.concat([train_data,test_data],axis = 0)
total_data = pd.get_dummies(total_data)
# train_data = pd.get_dummies(train_data)
# test_data = pd.get_dummies(test_data)
# 将两个数据集拼接到一起，然后做one-hot
# 然后再分开

print(total_data.shape)
train_data = total_data.iloc[0:h[0]+1]
test_data = total_data.iloc[h[0]+1:]
print('one-hot编码之后')
print(test_data.shape)
print(train_data.shape)
# imputer 用来填充缺失值
# 填补缺失值：sklearn.preprocessing.Imputer(
# missing_values=’NaN’,  缺失值为 NaN
# strategy=’mean’,      填充值为平均值
# axis=0,           指定轴数，默认axis = 0 代表列，axis = 1 代表行
# verbose=0,
# copy=True)        copy = True表示不在元数据集上修改，False，就地修改，即使设置成为False，也不会就地修改

my_imputer = Imputer()
train_data_with_imputed_values = my_imputer.fit_transform(train_data)
test_data_with_imputed_values = my_imputer.fit_transform(test_data)

X_scaled = preprocessing.normalize(train_data_with_imputed_values,norm = 'l2')
x_test_scaled = preprocessing.normalize(test_data_with_imputed_values,norm = 'l2')
print(X_scaled.shape)
print(x_test_scaled.shape)
print(target_train.shape)

"""
(1460, 267)
(1460,)
"""
#
# x = []
# y_train = []
# y_test = []
#
# # train_test_split 随机划分训练集和测试集
# # train_test_split(train_data,train_target,test_size=0.4, random_state=0)
# # test_size = 0.4 是样本占比
# # random_state 是随机数的种子
# # 返回值时训练集和测试集
# X_train,X_test,Y_train,Y_test = model_selection.train_test_split(X_scaled,train_data,random_state = 42)

x = []
y_train = []
y_test = []
# creating the data set for input to the randomforestregressor algorithm
X_train,X_test,Y_train,Y_test = model_selection.train_test_split(X_scaled,target_train,random_state=42)


"""
# 用一个循环来做观察合适的参数
for i in range(1,20):
    forest = RandomForestRegressor(n_estimators=100,max_depth=i ,random_state=2)
    forest.fit(X_train, Y_train)
    x.append(i)
    y_test.append(forest.score(X_test, Y_test))
    y_train.append(forest.score(X_train, Y_train))

plt.xlabel('Tree_depth')
plt.ylabel('Accuracy')
plt.plot(x,y_train,'red',x,y_test,'blue')
plt.show()
"""

forest = RandomForestRegressor(n_estimators=100,max_depth=5 ,random_state=42)
kfold = KFold(n_splits=5, shuffle=True, random_state=0)
scores = cross_val_score(forest, X_scaled, target_train,cv=kfold)
print("Cross-validation scores: {}".format(scores))
print('the mean is:{}'.format(np.mean(scores)))
forest.fit(X_train, Y_train)
print(X_train.shape,Y_train.shape,x_test_scaled.shape)
y = forest.predict(x_test_scaled)

submission_df = pd.DataFrame(data = {'Id':range(1461,1461+len(x_test_scaled)),'SalePrice':y})
print(submission_df.head(10))
submission_df.to_csv('submission_br.csv',columns = ['Id','SalePrice'],index = False)

s = pd.read_csv('submission_br.csv')
print(s.shape)
