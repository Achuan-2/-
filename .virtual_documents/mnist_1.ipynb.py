#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# 安装 TensorFlow
# 导入包
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)


mnist = tf.keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels)= mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0


train_images.shape


test_images.shape


train_labels.shape


train_labels[0]


# train_images[0]


plt.figure(figsize=(2,2))
plt.imshow(train_images[0],cmap=plt.cm.binary)
plt.colorbar()
plt.grid(False)
plt.show()


#显示训练集中的前 25 个图像，并在每个图像下方显示类名称。
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(True)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    # plt.imshow(train_images[i], cmap=plt.cm.ocean)
    plt.xlabel(train_labels[i])
plt.show()


model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.summary()


model.fit(train_images, train_labels, epochs=10)

model.evaluate(test_images,  test_labels, verbose=2)


# 如果直接进行预测
predictions = model.predict(test_images)
predictions[0]


# 在原有模型基础上附加一个 softmax  层，将 logits 转换成更容易理解的概率。
probability_model = tf.keras.Sequential([model,tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
predictions[0]


np.argmax(predictions[0])


# 图片绘制函数
def plot_image(i, predictions_array, label_array, img_array):
  predictions_array, true_label, img = predictions_array[i], label_array[i], img_array[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(predicted_label,
                                100*np.max(predictions_array),
                                true_label),
                                color=color)

# 概率绘制
def plot_value_array(i, predictions_array, label_array):
  predictions_array, true_label = predictions_array[i], label_array[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')


i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions, test_labels)
plt.show()


# 预测结果可视化
num_rows = 5
num_cols = 5
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
plt.tight_layout()
plt.show()


# 直接分析结果函数
def predict_result(i, predictions_array, label_array, img_array,right_array,error_array):
    predictions_array, true_label, img = predictions_array[i], label_array[i], img_array[i]
    predicted_label = np.argmax(predictions_array)
    message="序号{}\t 预测为{}\t概率{:2.0f}get_ipython().run_line_magic("\t实际是", " ({})\".format(i,predicted_label,")
                                100*np.max(predictions_array),true_label)

    if predicted_label == true_label:
        # print(message,"\t预测成功！")
        right_array.append(message)
    else:
        # print(message,"\t预测失败！")
        error_array.append(message)

right_array=[]
error_array=[]
num_images=10000
for i in range(num_images):
    predict_result(i, predictions, test_labels, test_images,right_array,error_array)


print(error_array)
print(len(error_array))


# 检查预测失败的图
i = 151
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions, test_labels)
plt.show()


i = 247
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions, test_labels)
plt.show()



model_c = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
  tf.keras.layers.MaxPooling2D((2, 2)),
  tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D((2, 2)),
  tf.keras.layers.Conv2D(14, (3, 3), activation='relu'),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(10,activation='softmax')
])



# 增加维度到4维
train_images_4 = train_images.reshape((60000, 28, 28, 1))
test_images_4 = test_images.reshape((10000, 28, 28, 1))


train_images.shape



model_c.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
history = model_c.fit(train_images_4, train_labels, epochs=5, 
                    validation_data=(test_images_4, test_labels))


plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()




test_loss, test_acc = model_c.evaluate(test_images_4,  test_labels, verbose=2)
print(test_acc)


# 将整个模型另存为 SavedModel。
get_ipython().getoutput("mkdir -p saved_model")
model.save('saved_model/my_model')
