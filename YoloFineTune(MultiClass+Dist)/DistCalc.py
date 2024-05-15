import cv2 as cv 
import numpy as np

# Distance constants 
KNOWN_DISTANCE = 45 #INCHES
SMALLc_WIDTH = 16 #INCHES
LARGEc_WIDTH = 3.0 #INCHES
CYLINDER_WIDTH = 9.0

# Object detector constant 
CONFIDENCE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.3

# colors for object detected
COLORS = [(255,0,0),(255,0,255),(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
GREEN =(0,255,0)
BLACK =(0,0,0)
# defining fonts 
FONTS = cv.FONT_HERSHEY_COMPLEX

# getting class names from classes.txt file 
class_names = []
with open("classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]
#  setttng up opencv net
yoloNet = cv.dnn.readNet('yolov4-tiny-custom_2000.weights', 'yolov4-tiny-custom.cfg')

yoloNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
yoloNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

model = cv.dnn_DetectionModel(yoloNet)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

# object detector funciton /method
def object_detector(image):
    classes, scores, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    # creating empty list to add objects data
    data_list =[]
    for (classid, score, box) in zip(classes, scores, boxes):
        # define color of each, object based on its class id 
        color= COLORS[int(classid) % len(COLORS)]
    
        label = "%s : %f" % (class_names[classid[0]], score)

        # draw rectangle on and label on object
        cv.rectangle(image, box, color, 2)
        #cv.putText(image, label, (box[0], box[1]-14), FONTS, 0.5, color, 2)
    
        # getting the data 
        # 1: class name  2: object width in pixels, 3: position where have to draw text(distance)
        if classid ==0: # person class id 
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2)])
        elif classid ==1:
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2)])
        elif classid ==2:
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2)])
        # if you want inclulde more classes then you have to simply add more [elif] statements here
        # returning list containing the object data. 
    return data_list

def focal_length_finder (measured_distance, real_width, width_in_rf):
    focal_length = (width_in_rf * measured_distance) / real_width

    return focal_length

# distance finder function 
def distance_finder(focal_length, real_object_width, width_in_frmae):
    distance = (real_object_width * focal_length) / width_in_frmae
    return distance

# reading the reference image from dir 
ref_SMALLc = cv.imread('image1.png')
ref_LARGEc = cv.imread('ReferenceImages/image4.png')
ref_CYLINDER = cv.imread('ReferenceImages/image4.png')

data1 = object_detector(ref_SMALLc)
D1 = data1[1][1]

data2 = object_detector(ref_LARGEc)
D2 = data2[0][1]

data3 = object_detector(ref_CYLINDER)
D3 = data3[0][1]

# finding focal length 
focal_CYLINDER = focal_length_finder(KNOWN_DISTANCE, CYLINDER_WIDTH, D3)
focal_SMALL_BOX = focal_length_finder(KNOWN_DISTANCE, SMALLc_WIDTH, D1)
focal_LARGE_BOX = focal_length_finder(KNOWN_DISTANCE, LARGEc_WIDTH, D2)


cap = cv.VideoCapture(0)
while True:
    ret, frame = cap.read()

    data = object_detector(frame) 
    for d in data:
        if d[0] =='Cylinder':
            distance = distance_finder(focal_SMALL_BOX, CYLINDER_WIDTH, d[1])
            x, y = d[2]
        elif d[0] =='Non traversable':
            distance = distance_finder (focal_LARGE_BOX, LARGEc_WIDTH, d[1])
            x, y = d[2]
        elif d[0] =='Traversable':
            distance = distance_finder (focal_SMALL_BOX, SMALLc_WIDTH, d[1])
            x, y = d[2]
        cv.rectangle(frame, (x, y-3), (x+150, y+23),BLACK,-1 )
        cv.putText(frame, f'Dis: {round(distance,2)} inch', (x+5,y+13), FONTS, 0.48, GREEN, 2)

    cv.imshow('frame',frame)
    
    key = cv.waitKey(1)
    if key ==ord('q'):
        break
cv.destroyAllWindows()
cap.release()
