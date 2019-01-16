import tensorflow as tf
import numpy as np
def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

def load_data(file_path):
    data=[]
    label=[]
    with open(file_path,"r") as data_file:
        raw_data=data_file.readlines()
        for raw_line in raw_data:
            sample=raw_line[:-1]
            sample=sample.split(",")
            label.append(  [(float(sample[-1]))]   )
            sample=np.array([float(feature) for feature in sample[:-1]])
            sample[-3]/=10
            sample[-2]/=100
            sample[-1]/=1000
            data.append(sample)
    return np.array(data),np.array(label)

sess = tf.InteractiveSession() 

#### 1.训练的数据
train_data,train_label=load_data("./train.data")
test_data,test_label=load_data("./test.data")
#print(train_label )

#### 2.定义节点准备接收数据
xs=tf.placeholder(tf.float32,shape=[None,57],name='x_input')
ys=tf.placeholder(tf.float32, shape=[None,1],name='y_input')

#### 3.定义神经层：隐藏层和预测层
l1=add_layer(xs,57,1,activation_function=None)
#### add output layer 输入值是隐藏层 l1，在预测层输出 1 个结果
prediction=add_layer(l1,1,1,activation_function=tf.nn.sigmoid)
#### 4.定义 loss 表达式
loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))
#### 5.选择 optimizer 使 loss 达到最小                   
#### 这一行定义了用什么方式去减少 loss，学习率是   
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

correct_prediction = tf.equal(tf.argmax(ys,1), tf.argmax(test_label,1))   #在测试阶段，测试准确度计算
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))                #多个批次的准确度均值

sess=tf.Session()

sess.run(tf.global_variables_initializer())
for i in range(len(train_data)):
    sess.run(train_step,feed_dict={xs:train_data,ys:train_label})
    if i % 50 == 0:
        print(sess.run(loss,feed_dict={xs:train_data,ys:train_label}))

    
for i in range(len(test_data)):
    sess.run(train_step,feed_dict={xs:test_data,ys:test_label})
    if i % 50 == 0:
        print(sess.run(correct_prediction,feed_dict={xs:test_data,ys:test_label}))
        print(sess.run(accuracy,feed_dict={xs:test_data,ys:test_label}))




