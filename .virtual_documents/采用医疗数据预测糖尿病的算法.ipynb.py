get_ipython().run_line_magic("matplotlib", " inline")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
diabetes = pd.read_csv('data/diabetes.csv')
diabetes.columns 
# 忽略警告
import warnings
warnings.filterwarnings("ignore")


diabetes.head()


print("Diabetes data set dimensions : {}".format(diabetes.shape))


diabetes.groupby('Outcome').size()


# 这一句学到了，这样就可以直接看每一列的分布情况了
diabetes.hist(figsize=(9, 9))


diabetes.isnull().sum()
diabetes.isna().sum()


print("Total : ", diabetes[diabetes.BloodPressure == 0].shape[0])
print(diabetes[diabetes.BloodPressure == 0].groupby('Outcome')['Age'].count())


diabetes_mod = diabetes[(diabetes.BloodPressure get_ipython().getoutput("= 0) & (diabetes.BMI != 0) & (diabetes.Glucose != 0)]")
print(diabetes_mod.shape)


diabetes_mod


X = diabetes_mod.iloc[:,:-1]
y = diabetes_mod.Outcome


from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


# 然后，将各个分类器按默认参数初始化，并建立一个模型列表。
models = []
models.append(('KNN', KNeighborsClassifier()))
models.append(('SVC', SVC()))
models.append(('LR', LogisticRegression()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('GNB', GaussianNB()))
models.append(('RF', RandomForestClassifier()))
models.append(('GB', GradientBoostingClassifier()))


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = diabetes_mod.Outcome, random_state=0)


# 然后我们用“accuracy_score”来计算各个模型的准确率。
names = []
scores = []
for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    scores.append(accuracy_score(y_test, y_pred))
    names.append(name)
tr_split = pd.DataFrame({'Name': names, 'Score': scores})
print(tr_split)


from sklearn.model_selection import KFold
names = []
scores = []
for name, model in models:
    
    kfold = KFold(n_splits=10, shuffle=True,random_state=10) 
    score = cross_val_score(model, X, y, cv=kfold, scoring='accuracy').mean()
    
    names.append(name)
    scores.append(score)
kf_cross_val = pd.DataFrame({'Name': names, 'Score': scores})
print(kf_cross_val)


axis.patches 


axis = sns.barplot(data = kf_cross_val,x = 'Name', y = 'Score' )
axis.set(xlabel='Classifier', ylabel='Accuracy')
for p in axis.patches: # 大概是返回每个柱子对象
    height = p.get_height() # 获取高度
    x = p.get_x() + p.get_width()/2 # 确定x坐标
    axis.text(x, height + 0.005, '{:1.4f}'.format(height), ha="center") 
plt.show()


from sklearn.model_selection import GridSearchCV


# Specify parameters
c_values = list(np.arange(1, 10))
param_grid = [
    {'C': c_values, 'penalty': ['l1'], 'solver' : ['liblinear'], 'multi_class' : ['ovr']},
   {'C': c_values, 'penalty': ['l2'], 'solver' : ['liblinear', 'newton-cg', 'lbfgs'], 'multi_class' : ['ovr']}
]


strat_k_fold=KFold(n_splits=10, shuffle=True,random_state=10) 


grid = GridSearchCV(LogisticRegression(), param_grid, cv=strat_k_fold, scoring='accuracy')
grid.fit(X, y)


print(grid.best_params_)
print(grid.best_estimator_)


logreg_new = LogisticRegression(C=3, multi_class='ovr', penalty='l2', solver='liblinear')
initial_score = cross_val_score(logreg_new, X, y, cv=strat_k_fold, scoring='accuracy').mean()
print("Final accuracy : {} ".format(initial_score))

