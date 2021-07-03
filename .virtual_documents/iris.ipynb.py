import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm, datasets
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
 #from sklearn.cross_validation import train_test_split  #������anaconda 3.6����ǰ�汾
from sklearn.model_selection import train_test_split #������anaconda 3.7���Ժ�汾


iris = datasets.load_iris()
# Bunch�������У�
# data���������顣
# target���ļ����ࡣ���β������ģ���filenamesһһ��ӦΪ0��1��2��
# target_names����ǩ���������Զ��壬Ĭ��Ϊ�ļ�������
# DESCR������������
# filenames���ļ�����


se = iris.data[0:50] # ɽ�β������50��
ve = iris.data[50:100] # ��ɫ�β������50��
vi = iris.data[100:150] # ά������������50��
se


comb = [[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]] #��άͼ��4�������������


plt.figure(figsize=(10,10))
for i in range(6):
    plt.subplot(231+i)
    plt.plot(se[:,comb[i][0]],se[:,comb[i][1]],'o',color="#ff0000")
    plt.plot(ve[:,comb[i][0]],ve[:,comb[i][1]],'^',color='#00ff00')
    plt.plot(vi[:,comb[i][0]],vi[:,comb[i][1]],'+',color='#ff00ff')
plt.show()



X_train,X_test,y_train,y_test = train_test_split(\
    iris['data'],iris['target'],random_state=0)

print("ѵ���������ݵĴ�С��{}".format(X_train.shape))
print("ѵ��������ǩ�Ĵ�С��{}".format(y_train.shape))
print("�����������ݵĴ�С��{}".format(X_test.shape))
print("����������ǩ�Ĵ�С��{}".format(y_test.shape))


knn= neighbors.KNeighborsClassifier(n_neighbors = 3)
# ѵ��ģ��
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)

# ����ģ��
print("ģ�;��ȣ�{:.2f}".format(np.mean(y_pred==y_test)))
print("ģ�;��ȣ�{:.2f}".format(knn.score(X_test,y_test)))



X_new = np.array([[1.1,5.9,1.4,2.2]])
prediction = knn.predict(X_new)
print("Ԥ���Ŀ������ǣ�{}".format(prediction))
print("Ԥ���Ŀ��������ǣ�{}".format(iris['target_names'][prediction]))


X = iris.data
y = iris.target
print(X.shape,y.shape)
# ����label_binarize���β���������ж�ֵ�����������β����������ת��Ϊ001��010��100�ĸ�ʽ��
y = label_binarize(y, classes=[0, 1, 2])
n_classes = y.shape[1]



#��X����800ά���������������ӷ����Ѷȡ�
random_state = np.random.RandomState(0) # α�����
n_samples, n_features = X.shape
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)] #ͨ��np.c_[ ]��ԭʼX����Ļ���������800ά����������


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=random_state)


# ����OneVsRestClassifierģ�齫��������ת��Ϊ����ķ�������Ӷ�����һ���µķ������������ķ�������ʹ��SVM��
classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True, random_state=random_state))


#���ǽ�ѵ��������������н������ѵ����ѵ�����֮�����ǽ����Լ��е��������������ȥ���Ӷ��õ����Լ���ÿ��������Ԥ�����y_score��
y_score = classifier.fit(X_train, y_train).decision_function(X_test)


#���������ֵ�precision��recall��average_precision��
precision = dict()
recall = dict()
average_precision = dict()


for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],  y_score[:, i])
    average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])

# �»���"_"�Ƿ��ص���ֵ����Ϊһ�����ƣ���ʱ"_"��Ϊ��ʱ�Ե�����ʹ�ã���ʾ������һ���ض������ƣ����ǲ������ں����ٴ��õ������ơ�


precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(),  y_score.ravel())
average_precision["micro"] = average_precision_score(y_test, y_score, average="micro")


# Plot Precision-Recall curve for each class
plt.style.use('seaborn') #ѡ��'seaborn'�����ָ�ʹ��ͼ����һ��
plt.clf()#clf �������������ǰͼ�񴰿�
plt.plot(recall["micro"], precision["micro"],
         label='micro-average Precision-recall curve (area = {0:0.2f})'.format(average_precision["micro"]))
for i in range(n_classes):
    plt.plot(recall[i], precision[i],
             label='Precision-recall curve of class {0} (area = {1:0.2f})'.format(i, average_precision[i]))
#xlim��ylim���ֱ�����X��Y�����ʾ��Χ��
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
#���ú����������
plt.xlabel('Recall', fontsize=16)
plt.ylabel('Precision',fontsize=16)
#����P-Rͼ�ı���
plt.title('Extension of Precision-Recall curve to multi-class',fontsize=16)
plt.legend(loc="lower right")#legend ����������ͼ���ĺ���
plt.show()



