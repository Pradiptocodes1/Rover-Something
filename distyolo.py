import cv2 as cv
import numpy as np

known_dist = 45
object_width=3.0

confidence_threshold=0.5
nms_threshold=0.5

colors=[(255,0,0),(255,0,255),(0,255,255),(255,255,0),(0,255,0),(255,0,0)]
green=(0,0,0)
black=(0,255,0)

fonts=cv.FONT_HERSHEY_COMPLEX

class_names=[]
with open("classes.txt","r") as f:
    class_names=[cname.strip() for cname in f.readlines()]

yoloNet=cv.dnn.readNet('yolov4-tiny-custom-training_best(red_updated).weights','yolov4-tiny-custom-training.cfg')

yoloNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
yoloNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

model=cv.dnn_DetectionModel(yoloNet)
model.setInputParams(size=(416,416), scale=1/255, swapRB=True)

def object_detector(image):
    classes, scores, boxes = model.detect(image, confidence_threshold, nms_threshold)
    data_list = []
    for(classid,score,box) in zip(classes,scores,boxes):
        color=colors[int(classid)%len(colors)]
        label="%s : %f" % (class_names[classid],score)

        cv.rectangle(image,box,color,2)
        cv.putText(image,label,(box[0],box[1]-14), fonts,0.5,color,2)

        if classid==0:
            data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
    return data_list

def focal_length_finder(measured_distance, real_width, width_in_rf):
    focal_length=(width_in_rf * measured_distance)/real_width
    return focal_length

def distance_finder(focal_length, real_object_width, width_in_frame):
    distance=(real_object_width*focal_length)/width_in_frame
    return distance

ref_obj=cv.imread('ReferenceImages\image10.png')

obj_data=object_detector(ref_obj)
obj_width_rf=obj_data[0][1]

focal_obj=focal_length_finder(known_dist, object_width, obj_width_rf)
cap=cv.VideoCapture(0)
while True:
    ret, frame = cap.read()
    data=object_detector(frame)
    for d in data:
        if d[0]=='cylinder':
            distance=distance_finder(focal_obj,object_width,d[1])
            x,y=d[2]
        cv.rectangle(frame,(x,y-3),(x+150,y+23),black,-1)
        cv.putText(frame,f'Dist: {round(distance,2)} inch',(x+5,y+13), fonts, 0.48, green, 2)

    cv.imshow('frame',frame)
    key=cv.waitKey(1)
    if key==ord('q'):
        break
cv.destroyAllWindows()
cap.release()
