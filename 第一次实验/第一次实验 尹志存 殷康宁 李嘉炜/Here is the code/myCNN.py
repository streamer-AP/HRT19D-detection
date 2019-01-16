#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

import numpy as np
import tensorflow as tf

# python pkl 文件读写
import pickle as pickle


tf.logging.set_verbosity(tf.logging.INFO)
'''
    作用：将 TensorFlow 日志信息输出到屏幕
    
    TensorFlow有五个不同级别的日志信息。其严重性为调试DEBUG<信息INFO<警告WARN<错误ERROR<致命FATAL。当你配置日志记录在任何级别，TensorFlow将输出与该级别相对应的所有日志消息以及更高程度严重性的所有级别的日志信息。例如，如果设置错误的日志记录级别，将得到包含错误和致命消息的日志输出，并且如果设置了调试级别，则将从所有五个级别获取日志消息。
    
    默认情况下，TENSFlow在WARN的日志记录级别进行配置，但是在跟踪模型训练时，需要将级别调整为INFO
'''

def cnn_model_fn(features, labels, mode):
    '''CNN函数模型'''
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # MNIST images are 28x28 pixels, and have one color channel
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
    '''图片规范化为28*28'''
    # Convolutional Layer #1
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 28, 28, 1]
    # Output Tensor Shape: [batch_size, 28, 28, 32]
    '''第一层'''
    '''
        作用
        
        2D 卷积层的函数接口 这个层创建了一个卷积核，将输入进行卷积来输出一个 tensor。如果 use_bias 是 True（且提供了 bias_initializer），则一个偏差向量会被加到输出中。最后，如果 activation 不是 None，激活函数也会被应用到输出中。
        
        参数
        
        inputs：Tensor 输入
        
        filters：整数，表示输出空间的维数（即卷积过滤器的数量）
        
        kernel_size：一个整数，或者包含了两个整数的元组/队列，表示卷积窗的高和宽。如果是一个整数，则宽高相等。
        
        strides：一个整数，或者包含了两个整数的元组/队列，表示卷积的纵向和横向的步长。如果是一个整数，则横纵步长相等。另外， strides 不等于1 和 dilation_rate 不等于1 这两种情况不能同时存在。
        
        padding："valid" 或者 "same"（不区分大小写）。"valid" 表示不够卷积核大小的块就丢弃，"same"表示不够卷积核大小的块就补0。 "valid" 的输出形状为
        
        "valid" 的输出形状为
        其中， 为输入的 size（高或宽）， 为 filter 的 size， 为 strides 的大小， 为向上取整。
        data_format：channels_last 或者 channels_first，表示输入维度的排序。
    '''
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 28, 28, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 32]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    '''
        池化
        目的是使得图片表示更小更可管理
        对每个activation map进行独立操作
    '''
    # Convolutional Layer #2
    # Computes 64 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 14, 14, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 64]
    '''第二层同理'''
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    # Pooling Layer #2
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 14, 14, 64]
    # Output Tensor Shape: [batch_size, 7, 7, 64]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 7, 7, 64]
    # Output Tensor Shape: [batch_size, 7 * 7 * 64]
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 7 * 7 * 64]
    # Output Tensor Shape: [batch_size, 1024]
    '''
        dense ：全连接层  相当于添加一个层
    '''
    dense = tf.layers.dense(
        inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    # Add dropout operation; 0.6 probability that element will be kept
    '''
        训练时丢弃一些节点，inputs是输入层,rate是丢弃比例，training是个bool，我额外加了括号方便理解
    '''
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=(mode == tf.estimator.ModeKeys.TRAIN))
    # Logits layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 30]
    logits = tf.layers.dense(inputs=dropout, units=30)
    '''
       定义一个字典，键值对为 类型：可能性
    '''
    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    
    '''
        训练/测试模式下不同动作
    '''
    '''
        这个类被用来训练和评估tensorflow模型
        这个对象封装了EstimatorSpe类，根据给定的输入和一些其他的参数，去训练或者评估模型。
        这个类所有的输出都会被写到”model_dir”参数所对应的目录，如果这个参数为null，则会被写入到一个临时文件夹。
        Estimator类中config参数需传递一个RunConfig对象实例，这个对象用来控制程序的运行环境。他会被传入到Model实例中。
    '''
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    '''
        交叉损失熵？总之这是一个评估指标嗯。
    '''
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    '''
        随机梯度下降法优化网络，学习率0.001（我试图改这个参数证明“我能掌握这个”，但是调过后模型崩了......）
    '''
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    '''
        定义评估准确度的字典
    '''
    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
        # Load training and eval data
        # mnist = tf.contrib.learn.datasets.load_dataset("mnist")
        #   mnist = tf.contrib.learn.datasets.load_dataset("mnist")
        #   from tensorflow.examples.tutorials.mnist import input_data
        #   mnist = input_data.read_data_sets("Controller/MNIST_data/", one_hot=True) #MNIST数据输入

        #   train_data = mnist.train.images  # Returns np.array
        #   train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
        #   eval_data = mnist.test.images  # Returns np.array
        #   eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    '''加载训练集以及测试集'''
    train_data = np.array(pickle.load(open('Model/train_data.plk', 'rb')) )
    train_labels = np.array(pickle.load(open('Model/train_labels.plk', 'rb')) )
    eval_data = np.array(pickle.load(open('Model/eval_data.plk', 'rb')) )
    eval_labels = np.array(pickle.load(open('Model/eval_labels.plk', 'rb')) )

    # with tf.Session() as sess:
    #     train_data = tf.convert_to_tensor(train_data_np)
    #     eval_data = tf.convert_to_tensor(eval_data_np)

    # print(train_data)
    # print(train_labels)
    # print(eval_data)
    # print(eval_labels)cls
    # Create the Estimator
    '''
        定义归类器，训练模型，这个mnist和tf官网那个没什么关系啊......
    '''
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="mnist_convnet_model")
    
    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    '''每训练50次的时候输出预测的概率值'''
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    '''
        定义训练输入
    '''
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=20,
        num_epochs=None,
        shuffle=True)
    '''训练500步后停止，利用hook参数触发日志函数'''
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=500,
        hooks=[logging_hook])

    # Evaluate the model and print results
    '''评估模型，shuffle=False说明循环遍历数据'''
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


if __name__ == "__main__":
    tf.app.run()
