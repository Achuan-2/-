import pandas as pd
import numpy as np
get_ipython().run_line_magic("matplotlib", " inline")
import numpy as np
from scipy import stats
mush_df = pd.read_csv('data/mushrooms.csv')



mush_df


mush_df[mush_df["class"]=='p']


mush_df[mush_df["class"]=='e']


mush_df_encoded=pd.get_dummies(mush_df)


mush_df_encoded


# 将特征和类别标签分别赋值给 X 和 y，因变量y选择第二列（1有毒，0无毒）

X_mush = mush_df_encoded.iloc[:,2:]#所有行、第三列及往后
y_mush = mush_df_encoded.iloc[:,1]#所有行、第二列（第二列是‘有毒’，1表示有毒，0表示无毒


from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
# pca = PCA(n_components=117, whiten=True, random_state=42)



# model = make_pipeline(pca, svc)


from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X_mush, y_mush,test_size=0.5, random_state=41)


from sklearn.model_selection import GridSearchCV
# param_grid = {'C':[1,2,3,4,5,6,10]}#设置的C可能的值是1，2，5，10，可以自由设置
param_grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
]
svc = SVC(class_weight='balanced')
grid = GridSearchCV(svc,param_grid,cv=10,verbose=1,n_jobs=8)
get_ipython().run_line_magic("time", " grid.fit(Xtrain, ytrain)")
print('model.best_params_::  ',grid.best_params_)
print('model.best_score_:: ',grid.best_score_)
model=grid.best_estimator_


Xtest.shape



import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
test_predict_label  = model.decision_function(Xtest)
fpr,tpr,threshold = roc_curve(ytest, test_predict_label,pos_label=1)
roc_auc = auc(fpr,tpr)

threshold 


print(fpr)
print(tpr)
print(threshold)


# ytest


 model.decision_function(Xtest)


plt.figure()
lw = 2
plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = get_ipython().run_line_magic("0.2f)'", " % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线")
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


from sklearn.metrics import classification_report
print(classification_report(ytest, yfit,target_names=['e','p']))


y = np.array([-1,-1,1,1])
pred = np.array([0.1,0.4,0.35,0.8])
def Find_Optimal_Cutoff(TPR, FPR, threshold):
    y = TPR - FPR
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point
def ROC(label, y_prob):
    """
    Receiver_Operating_Characteristic, ROC
    :param label: (n, )
    :param y_prob: (n, )
    :return: fpr, tpr, roc_auc, optimal_th, optimal_point
    """
    fpr, tpr, thresholds = roc_curve(label, y_prob)
    roc_auc = auc(fpr, tpr)
    optimal_th, optimal_point = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thresholds)
    return fpr, tpr, roc_auc, optimal_th, optimal_point
def ROC_plot(label, y_prob):
    fpr, tpr, roc_auc, optimal_th, optimal_point = ROC(label, y_prob)
    plt.figure(1)
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.plot(optimal_point[0], optimal_point[1], marker='o', color='r')
    plt.text(optimal_point[0], optimal_point[1], f'Threshold:{optimal_th:.2f}')
    plt.title("ROC-AUC")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.show()
ROC_plot(y, pred)
    


clf = SVC(kernel='linear', class_weight='balanced',C=0.1)
clf.fit(Xtrain, ytrain)
yfit = clf.predict(Xtest)
print(classification_report(ytest, yfit,digits=4,output_dict=True))
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(ytest, yfit)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label');










model.predict(Xtest)



