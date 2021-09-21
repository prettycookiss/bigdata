# import tensorflow as tf
# import numpy as np
#
# # from tensorflow import keras
# # import  matplotlib.pyplot as plt
# #
# # #加载数据
# # fashion_mist = keras.datasets.fashion_mnist
# # #训练集和测试集的划分
# # (train_images, train_lables), (test_images, test_lables) = fashion_mist.load_data()
# #
# # # 打印图像
# # plt.figure()
# # plt.xticks()
# # plt.yticks()
# # plt.imshow(train_images[0])
# # plt.show()
#


# 导入模块
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

#加载数据
fashion_mnist = keras.datasets.fashion_mnist
# 进行训练集和测试集的划分
(train_images,train_labels),(test_images,test_labels)=fashion_mnist.load_data()
print(train_images.shape)
# 将数据进行预处理，将每个像素点压缩在0和1之间
train_images = train_images/255.0
test_images = test_images/255.0
# 搭建简单地神经网络
def create_model():
    model = tf.keras.models.Sequential([
        keras.layers.Flatten(input_shape=(28,28)),
        keras.layers.Dense(128,activation="relu"),
        keras.layers.Dense(10) ])
#     编译模型
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits="True"),
        metrics=[tf.metrics.SparseTopKCategoricalAccuracy()]
    )

#     返回模型
    return model

# 构建模型
new_model = create_model()

# 训练模型
new_model.fit(train_images,train_labels,epochs=30)

# 保存模型
new_model.save("model/my_model13.h5")





# 导入模块
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

#加载数据
fashion_mnist = keras.datasets.fashion_mnist
# 进行训练集和测试集的划分
(train_images,train_labels),(test_images,test_labels)=fashion_mnist.load_data()
print(train_images.shape)
# 将数据进行预处理，将每个像素点压缩在0和1之间
train_images = train_images/255.0
test_images = test_images/255.0
# 搭建简单地神经网络
def create_model():
    model = tf.keras.models.Sequential([
        keras.layers.Flatten(input_shape=(28,28)),
        keras.layers.Dense(128,activation="relu"),
        keras.layers.Dense(10) ])
#     编译模型
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits="True"),
        metrics=[tf.metrics.SparseTopKCategoricalAccuracy()]
    )

#     返回模型
    return model

# 构建模型
new_model = create_model()

# 训练模型
new_model.fit(train_images,train_labels,epochs=30)

# 保存模型
new_model.save("model/my_model13.h5")




# 导入模块
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

#加载数据
fashion_mnist = keras.datasets.fashion_mnist
# 进行训练集和测试集的划分
(train_images,train_labels),(test_images,test_labels)=fashion_mnist.load_data()
print(train_images.shape)
# 将数据进行预处理，将每个像素点压缩在0和1之间
train_images = train_images/255.0
test_images = test_images/255.0
# 搭建简单地神经网络
def create_model():
    model = tf.keras.models.Sequential([
        keras.layers.Flatten(input_shape=(28,28)),
        keras.layers.Dense(128,activation="relu"),
        keras.layers.Dense(10) ])
#     编译模型
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits="True"),
        metrics=[tf.metrics.SparseTopKCategoricalAccuracy()]
    )

#     返回模型
    return model

# 构建模型
new_model = create_model()

# 训练模型
new_model.fit(train_images,train_labels,epochs=30)

# 保存模型
new_model.save("model/my_model13.h5")



# 导入模块
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

#加载数据
fashion_mnist = keras.datasets.fashion_mnist
# 进行训练集和测试集的划分
(train_images,train_labels),(test_images,test_labels)=fashion_mnist.load_data()
print(train_images.shape)
# 将数据进行预处理，将每个像素点压缩在0和1之间
train_images = train_images/255.0
test_images = test_images/255.0
# 搭建简单地神经网络
def create_model():
    model = tf.keras.models.Sequential([
        keras.layers.Flatten(input_shape=(28,28)),
        keras.layers.Dense(128,activation="relu"),
        keras.layers.Dense(10) ])
#     编译模型
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits="True"),
        metrics=[tf.metrics.SparseTopKCategoricalAccuracy()]
    )

#     返回模型
    return model

# 构建模型
new_model = create_model()

# 训练模型
new_model.fit(train_images,train_labels,epochs=30)

# 保存模型
new_model.save("model/my_model13.h5")









#导入模块
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

#导入模块
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

#使图像显示中文
plt.rcParams["font.sans-serif"]=["FangSong"]

#加载数据
fashion_mist = keras.datasets.fashion_mnist
(train_images, train_lables), (test_images, test_lables) = fashion_mist.load_data()

# 构建标签列表
class_names = ['T恤/上衣', '裤子', '套衫', '连衣裙', '外套', '凉鞋', '衬衫', '运动鞋', '手提包', '踝靴']

#数据预处理,将每个像素点压缩在0和1之间
train_images = train_images/255.0
test_images = test_images/255.0

#加载神经网络模型
new_model = keras.models.load_model("model/my_model13.h5")

#对刚才训练的模型进行测试
test_loss,test_acc = new_model.evaluate(test_images,test_lables,verbose = 2)
print("\nTest accuracy:{:5.2f}%".format(100 * test_acc))

#对数据进行预测
probability_model = tf.keras.Sequential([new_model,tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
print(predictions[6])

# 答应所有的预测结果
plt.figure()
plt.subplot(1,2,1)
plt.xticks([])
plt.yticks([])
plt.imshow(test_images[0],cmap=plt.cm.binary)
plt.xlabel("{} 预测正确率：{:2.0f}%".format(class_names[np.argmax(predictions[6])],
                               100*np.max(predictions[6])),fontsize=20,color="blue")
plt.subplot(1,2,2)
plt.xticks(range(10),class_names)
plt.yticks([])
thisplot = plt.bar(range(10), predictions[6], color="#777777")
plt.ylim([0, 1])
predicted_label = np.argmax(predictions[6])

thisplot[predicted_label].set_color('blue')
# thisplot[true_label].set_color('blue')
plt.show()
print("模型预测的结果为：{}".format(class_names[np.argmax(predictions[6])]))




#导入模块
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

#导入模块
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

#使图像显示中文
plt.rcParams["font.sans-serif"]=["FangSong"]

#加载数据
fashion_mist = keras.datasets.fashion_mnist
(train_images, train_lables), (test_images, test_lables) = fashion_mist.load_data()

# 构建标签列表
class_names = ['T恤/上衣', '裤子', '套衫', '连衣裙', '外套', '凉鞋', '衬衫', '运动鞋', '手提包', '踝靴']

#数据预处理,将每个像素点压缩在0和1之间
train_images = train_images/255.0
test_images = test_images/255.0

#加载神经网络模型
new_model = keras.models.load_model("model/my_model13.h5")

#对刚才训练的模型进行测试
test_loss,test_acc = new_model.evaluate(test_images,test_lables,verbose = 2)
print("\nTest accuracy:{:5.2f}%".format(100 * test_acc))

#对数据进行预测
probability_model = tf.keras.Sequential([new_model,tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
print(predictions[6])

# 答应所有的预测结果
plt.figure()
plt.subplot(1,2,1)
plt.xticks([])
plt.yticks([])
plt.imshow(test_images[0],cmap=plt.cm.binary)
plt.xlabel("{} 预测正确率：{:2.0f}%".format(class_names[np.argmax(predictions[6])],
                               100*np.max(predictions[6])),fontsize=20,color="blue")
plt.subplot(1,2,2)
plt.xticks(range(10),class_names)
plt.yticks([])
thisplot = plt.bar(range(10), predictions[6], color="#777777")
plt.ylim([0, 1])
predicted_label = np.argmax(predictions[6])

thisplot[predicted_label].set_color('blue')
# thisplot[true_label].set_color('blue')
plt.show()
print("模型预测的结果为：{}".format(class_names[np.argmax(predictions[6])]))






#导入模块
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

#导入模块
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

#使图像显示中文
plt.rcParams["font.sans-serif"]=["FangSong"]

#加载数据
fashion_mist = keras.datasets.fashion_mnist
(train_images, train_lables), (test_images, test_lables) = fashion_mist.load_data()

# 构建标签列表
class_names = ['T恤/上衣', '裤子', '套衫', '连衣裙', '外套', '凉鞋', '衬衫', '运动鞋', '手提包', '踝靴']

#数据预处理,将每个像素点压缩在0和1之间
train_images = train_images/255.0
test_images = test_images/255.0

#加载神经网络模型
new_model = keras.models.load_model("model/my_model13.h5")

#对刚才训练的模型进行测试
test_loss,test_acc = new_model.evaluate(test_images,test_lables,verbose = 2)
print("\nTest accuracy:{:5.2f}%".format(100 * test_acc))

#对数据进行预测
probability_model = tf.keras.Sequential([new_model,tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
print(predictions[6])

# 答应所有的预测结果
plt.figure()
plt.subplot(1,2,1)
plt.xticks([])
plt.yticks([])
plt.imshow(test_images[0],cmap=plt.cm.binary)
plt.xlabel("{} 预测正确率：{:2.0f}%".format(class_names[np.argmax(predictions[6])],
                               100*np.max(predictions[6])),fontsize=20,color="blue")
plt.subplot(1,2,2)
plt.xticks(range(10),class_names)
plt.yticks([])
thisplot = plt.bar(range(10), predictions[6], color="#777777")
plt.ylim([0, 1])
predicted_label = np.argmax(predictions[6])

thisplot[predicted_label].set_color('blue')
# thisplot[true_label].set_color('blue')
plt.show()
print("模型预测的结果为：{}".format(class_names[np.argmax(predictions[6])]))






#导入模块
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

#导入模块
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

#使图像显示中文
plt.rcParams["font.sans-serif"]=["FangSong"]

#加载数据
fashion_mist = keras.datasets.fashion_mnist
(train_images, train_lables), (test_images, test_lables) = fashion_mist.load_data()

# 构建标签列表
class_names = ['T恤/上衣', '裤子', '套衫', '连衣裙', '外套', '凉鞋', '衬衫', '运动鞋', '手提包', '踝靴']

#数据预处理,将每个像素点压缩在0和1之间
train_images = train_images/255.0
test_images = test_images/255.0

#加载神经网络模型
new_model = keras.models.load_model("model/my_model13.h5")

#对刚才训练的模型进行测试
test_loss,test_acc = new_model.evaluate(test_images,test_lables,verbose = 2)
print("\nTest accuracy:{:5.2f}%".format(100 * test_acc))

#对数据进行预测
probability_model = tf.keras.Sequential([new_model,tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
print(predictions[6])

# 答应所有的预测结果
plt.figure()
plt.subplot(1,2,1)
plt.xticks([])
plt.yticks([])
plt.imshow(test_images[0],cmap=plt.cm.binary)
plt.xlabel("{} 预测正确率：{:2.0f}%".format(class_names[np.argmax(predictions[6])],
                               100*np.max(predictions[6])),fontsize=20,color="blue")
plt.subplot(1,2,2)
plt.xticks(range(10),class_names)
plt.yticks([])
thisplot = plt.bar(range(10), predictions[6], color="#777777")
plt.ylim([0, 1])
predicted_label = np.argmax(predictions[6])

thisplot[predicted_label].set_color('blue')
# thisplot[true_label].set_color('blue')
plt.show()
print("模型预测的结果为：{}".format(class_names[np.argmax(predictions[6])]))






#导入模块
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

#导入模块
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

#使图像显示中文
plt.rcParams["font.sans-serif"]=["FangSong"]

#加载数据
fashion_mist = keras.datasets.fashion_mnist
(train_images, train_lables), (test_images, test_lables) = fashion_mist.load_data()

# 构建标签列表
class_names = ['T恤/上衣', '裤子', '套衫', '连衣裙', '外套', '凉鞋', '衬衫', '运动鞋', '手提包', '踝靴']

#数据预处理,将每个像素点压缩在0和1之间
train_images = train_images/255.0
test_images = test_images/255.0

#加载神经网络模型
new_model = keras.models.load_model("model/my_model13.h5")

#对刚才训练的模型进行测试
test_loss,test_acc = new_model.evaluate(test_images,test_lables,verbose = 2)
print("\nTest accuracy:{:5.2f}%".format(100 * test_acc))

#对数据进行预测
probability_model = tf.keras.Sequential([new_model,tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
print(predictions[6])

# 答应所有的预测结果
plt.figure()
plt.subplot(1,2,1)
plt.xticks([])
plt.yticks([])
plt.imshow(test_images[0],cmap=plt.cm.binary)
plt.xlabel("{} 预测正确率：{:2.0f}%".format(class_names[np.argmax(predictions[6])],
                               100*np.max(predictions[6])),fontsize=20,color="blue")
plt.subplot(1,2,2)
plt.xticks(range(10),class_names)
plt.yticks([])
thisplot = plt.bar(range(10), predictions[6], color="#777777")
plt.ylim([0, 1])
predicted_label = np.argmax(predictions[6])

thisplot[predicted_label].set_color('blue')
# thisplot[true_label].set_color('blue')
plt.show()
print("模型预测的结果为：{}".format(class_names[np.argmax(predictions[6])]))






#导入模块
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

#导入模块
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

#使图像显示中文
plt.rcParams["font.sans-serif"]=["FangSong"]

#加载数据
fashion_mist = keras.datasets.fashion_mnist
(train_images, train_lables), (test_images, test_lables) = fashion_mist.load_data()

# 构建标签列表
class_names = ['T恤/上衣', '裤子', '套衫', '连衣裙', '外套', '凉鞋', '衬衫', '运动鞋', '手提包', '踝靴']

#数据预处理,将每个像素点压缩在0和1之间
train_images = train_images/255.0
test_images = test_images/255.0

#加载神经网络模型
new_model = keras.models.load_model("model/my_model13.h5")

#对刚才训练的模型进行测试
test_loss,test_acc = new_model.evaluate(test_images,test_lables,verbose = 2)
print("\nTest accuracy:{:5.2f}%".format(100 * test_acc))

#对数据进行预测
probability_model = tf.keras.Sequential([new_model,tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
print(predictions[6])

# 答应所有的预测结果
plt.figure()
plt.subplot(1,2,1)
plt.xticks([])
plt.yticks([])
plt.imshow(test_images[0],cmap=plt.cm.binary)
plt.xlabel("{} 预测正确率：{:2.0f}%".format(class_names[np.argmax(predictions[6])],
                               100*np.max(predictions[6])),fontsize=20,color="blue")
plt.subplot(1,2,2)
plt.xticks(range(10),class_names)
plt.yticks([])
thisplot = plt.bar(range(10), predictions[6], color="#777777")
plt.ylim([0, 1])
predicted_label = np.argmax(predictions[6])

thisplot[predicted_label].set_color('blue')
# thisplot[true_label].set_color('blue')
plt.show()
print("模型预测的结果为：{}".format(class_names[np.argmax(predictions[6])]))








