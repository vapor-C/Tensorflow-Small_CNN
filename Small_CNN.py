#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 19:17:55 2017

@author: mml
"""
# tensorflow的MNIST数据加载模块
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_sets("MNIST_data/",one_hot = True)
# 将session注册为默认session
sess = tf.InteractiveSession()

# 后面网络会有较多的参数，这里定义一个初始化函数以便重复使用
def weight_variable(shape):
    # 随机的截断正态分布，标准差为0.1
    initial = tf.truncated_normal(shape,stddev = 0.1)
    return tf.Variable(initial)
    
def bias_variable(shape):
    # 偏置为小的正值0.1
    initial = tf.constant(0.1,shape = shape)
    return tf.Variable(initial)
# 卷积层和池化层也是需要重复使用的，因此也为他们创建函数   
def conv2d(x,w):
    # tf.nn.conv2d是tensorflow内置的卷积函数
    # x为输入，w是卷积参数
    # strides表示模板移动步长，全1表示不遗漏的划过图片每一个点
    # padding代表边界的处理方式 same表示卷积输入和输出同样尺寸
    return tf.nn.conv2d(x,w,strides = [1,1,1,1],padding = 'SAME')

def max_pool_2x2(x):
    # tf.nn.max_pool是tensorflow的最大池化函数
    return tf.nn.max_pool(x,ksize = [1,2,2,1],strides = [1,2,2,1],padding = 'SAME')
    
   
x = tf.placeholder(tf.float32,[None,784])
y_ = tf.placeholder(tf.float32,[None,10])
# 卷积神经网络会用到空间结构信息
# 1D的1*784的输入向量转为原始的2D 28*28
# reshape是tensorflow变形函数
# -1表示样本数量不固定，1表示颜色通道
x_image = tf.reshape(x,[-1,28,28,1])

# 第一个卷积层
# [5,5,1,32]表示5*5卷积模板，1个通道，32个卷积模板
w_conv1 = weight_variable([5,5,1,32])
# 每个模板一个偏置
b_conv1 = bias_variable([32])
# 卷积
h_conv1 = tf.nn.relu(conv2d(x_image,w_conv1)+b_conv1)
# 池化
h_pool1 = max_pool_2x2(h_conv1)

# 第二个卷积层
# 卷积大小不变，通道变为32，卷积模板变为64
w_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1,w_conv2)+b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 全连接层（实际就是一个MLP）
# 28*28经过两次2*2pooling变为7*7，所以tensor变为7*7*64
# 经过一个1024个节点的隐含层
w_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])
# 将之前的2D向量reshape为1D
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1)+b_fc1)
# 隐含层输出加个dropout防止过拟合
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)
# 输出层sofamax为10类
w_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,w_fc2)+b_fc2)

# cost
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y_conv),reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.arg_max(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

tf.global_variables_initializer().run()
# 最后batch feed数据进行训练
# 每100次训练计算一次准确率
# 训练时keepprob为0.5，测试时keepprob为1
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0})
        print "step %d,training accuracy %g"%(i,train_accuracy)
    train_step.run(feed_dict = {x:batch[0],y_:batch[1],keep_prob:0.5})

print "test accuracy %g"%accuracy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0})