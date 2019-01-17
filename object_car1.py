import sys

#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import cv2

import numpy as np

from os.path import join



carpath = "/home/wx/mypy/detection/car/"

nocarpath = "/home/wx/mypy/detection/nocar/"

def path(path,i):

    if i<9:

        return "%s/0000%d.jpg"  % (path,i+1)

    if 9<=i<99:

        return "%s/000%d.jpg"  % (path,i+1)

    if 99<=i<999:

        return "%s/00%d.jpg"  % (path,i+1)

 



 #

detect = cv2.xfeatures2d.SIFT_create()#提取关键点

extract = cv2.xfeatures2d.SIFT_create()#提取特征

#FLANN匹配器有两个参数：

flann_params = dict(algorithm = 1, trees = 5)#1为FLANN_INDEX_KDTREE

matcher = cv2.FlannBasedMatcher(flann_params, {})#匹配特征

#创建bow训练器，簇数为40

bow_kmeans_trainer = cv2.BOWKMeansTrainer(40)

#初始化bow提取器

extract_bow = cv2.BOWImgDescriptorExtractor(extract, matcher)

 

def extract_sift(path):#参数为路径

  im = cv2.imread(path,0)

  return extract.compute(im, detect.detect(im))[1]#返回描述符

#读取8个正样本和8个负样本

for i in range(8):

  bow_kmeans_trainer.add(extract_sift(path(carpath,i)))

  bow_kmeans_trainer.add(extract_sift(path(nocarpath,i)))

#利用训练器的cluster（）函数，执行k-means分类并返回词汇

#k-means：

voc = bow_kmeans_trainer.cluster()

extract_bow.setVocabulary( voc )

 

def bow_features(path):

  im = cv2.imread(path,0)

  return extract_bow.compute(im, detect.detect(im))

#两个数组，分别为训练数据和标签，并用bow提取器产生的描述符填充

traindata, trainlabels = [],[]

#traindata.extend(bow_features(path(3))); trainlabels.append(1)

for i in range(8): 

  traindata.extend(bow_features(path(carpath,i))); trainlabels.append(1)#1为正匹配

  traindata.extend(bow_features(path(nocarpath,i))); trainlabels.append(-1)#-1为负匹配

#创建SVM实例，

svm = cv2.ml.SVM_create()

svm.train(np.array(traindata), cv2.ml.ROW_SAMPLE, np.array(trainlabels))

 

def predict(path):

  f = bow_features(path);  

  p = svm.predict(f)

  print (path, "\t", p[1][0][0])

  return p

#预测结果 

car ,notcar = "/home/wx/1.jpg", "/home/wx/2.png"

car_img = cv2.imread(car)

notcar_img = cv2.imread(notcar)

car_predict = predict(car)

not_car_predict = predict(notcar)



font = cv2.FONT_HERSHEY_SIMPLEX

 

if (car_predict[1][0][0] == 1.0):#predict结果为1.0表示能检测到汽车

  cv2.putText(car_img,'Car Detected',(10,30), font, 1,(0,255,0),2,cv2.LINE_AA)

 

if (not_car_predict[1][0][0] == -1.0):#predict结果为-1.0表示不能检测到汽车

  cv2.putText(notcar_img,'Car Not Detected',(10,30), font, 1,(0,0, 255),2,cv2.LINE_AA)

 

cv2.imshow('BOW + SVM Success', car_img)

cv2.imshow('BOW + SVM Failure', notcar_img)

cv2.waitKey(0)

cv2.destroyAllWindows()
