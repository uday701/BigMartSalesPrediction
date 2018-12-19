import pandas as pd
import seaborn as sns
import matplotlib as plt
import numpy as np

train=pd.read_csv('Train.csv')
test=pd.read_csv('Test.csv')

train['source']='train'
test['source']='test'
data=pd.concat([train,test],ignore_index=True)

item_avg_weight = data.pivot_table(values='Item_Weight', index='Item_Identifier')
print(item_avg_weight.head())
weight_null_index=data['Item_Weight'].isnull()
data.loc[weight_null_index,'Item_Weight']=data.loc[weight_null_index,'Item_Identifier'].apply(lambda x:item_avg_weight.Item_Weight[x])

from scipy.stats import mode
item_avg_mode=data.dropna(subset=['Outlet_Size']).pivot_table(values='Outlet_Size',index='Outlet_Type',aggfunc=(lambda x:mode(x).mode[0]))
null_outlet_size=data['Outlet_Size'].isnull()
data.loc[null_outlet_size,'Outlet_Size']=data.loc[null_outlet_size,'Outlet_Type'].apply(lambda x:item_avg_mode.Outlet_Size[x])

avg_visibility=data.pivot_table(values='Item_Visibility',index='Item_Identifier')
zero_visibility=(data['Item_Visibility']==0)
data.loc[zero_visibility,'Item_Visibility']=data.loc[zero_visibility,'Item_Identifier'].apply(lambda x:avg_visibility.Item_Visibility[x])

data['combined_item_type']=data['Item_Identifier'].apply(lambda x:x[0:2])
data['combined_item_type']=data['combined_item_type'].map({'FD':'Food','NC':'Non-Consumable','DR':'Drinks'})

data['Years']=2013-data.Outlet_Establishment_Year
data.Item_Fat_Content = data.Item_Fat_Content.map({'Low Fat':'Low Fat','Regular':'Regular','low fat':'Low Fat','LF':'Low Fat','reg':'Regular'})

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['Outlet'] = le.fit_transform(data['Outlet_Identifier'])

var = ['Item_Fat_Content','Outlet_Location_Type','combined_item_type','Outlet_Size','Outlet_Type','Outlet']
for x in var:
    data[x]=le.fit_transform(data[x])

data = pd.get_dummies(data,columns=var)

data.drop(['Item_Type','Outlet_Establishment_Year'],axis=1,inplace=True)

train = data.loc[data['source']=="train"]
test = data.loc[data['source']=="test"]

test.drop(['Item_Outlet_Sales','source'],axis=1,inplace=True)
train.drop(['source'],axis=1,inplace=True)

target = 'Item_Outlet_Sales'
IDcol = ['Item_Identifier','Outlet_Identifier']

from sklearn import model_selection,metrics
def modelfit(alg, dtrain, dtest, predictors, target, IDcol, filename):
    alg.fit(dtrain[predictors], dtrain[target])
    dtrain_predictions = alg.predict(dtrain[predictors])
    cv_score = model_selection.cross_val_score(alg, dtrain[predictors], dtrain[target], cv=20,)
    cv_score = np.sqrt(np.abs(cv_score))
    print ("\nModel Report")
    print ("RMSE : %.4g" % np.sqrt(metrics.mean_squared_error(dtrain[target].values, dtrain_predictions)))
    print("CV Score : Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
    dtest[target] = alg.predict(dtest[predictors])
    IDcol.append(target)
    submission = pd.DataFrame({ x: dtest[x] for x in IDcol})
    submission.to_csv(filename, index=False)

from sklearn.linear_model import LinearRegression, Ridge, Lasso
predictors = [x for x in train.columns if x not in [target]+IDcol]
# print predictors
alg1 = LinearRegression(normalize=True)
modelfit(alg1, train, test, predictors, target, IDcol, 'alg1.csv')
coef1 = pd.Series(alg1.coef_, predictors).sort_values()
coef1.plot(kind='bar', title='Model Coefficients')

coef1

predictors = [x for x in train.columns if x not in [target]+IDcol]
alg2 = Ridge(alpha=0.05,normalize=True)
modelfit(alg2, train, test, predictors, target, IDcol, 'alg2.csv')
coef2 = pd.Series(alg2.coef_, predictors).sort_values()
coef2.plot(kind='bar', title='Model Coefficients')