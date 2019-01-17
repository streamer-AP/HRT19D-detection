
#导入了 ImageAI 目标检测类

from imageai.Detection import ObjectDetection

import os



execution_path = os.getcwd()



#定义了目标检测类

detector = ObjectDetection()

#模型的类型设置为 RetinaNet

detector.setModelTypeAsRetinaNet()

#确定模型路径

detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))

#导入模型

detector.loadModel()

#调用目标检测函数

detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "cars_test/00008.jpg"), output_image_path=os.path.join(execution_path , "imagenew1.jpg"))



#打印出所检测到的每个目标的名称及其概率值

for eachObject in detections:

     print(eachObject["name"] + " : " + eachObject["percentage_probability"] )
