import cv2 as cv
import time

confidence_threshold=0.5
nms_threshold=0.5

colors=[(0,255,255),(255,255,0),(0,255,0),(255,0,0)]
green=(0,255,0)
red=(0,0,255)
pink=(147,20,255)
orange=(0,69,255)
fonts=cv.FONT_HERSHEY_COMPLEX

class_names=[]
with open("classes.txt","r") as f:
    class_names=[cname.strip() for cname in f.readlines()]
yoloNet=cv.dnn.readNet('yolov4-tiny-custom-training_best(red_updated).weights','yolov4-tiny-custom-training.cfg') #weights are custom trained.
yoloNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
yoloNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

model=cv.dnn_DetectionModel(yoloNet)
model.setInputParams(size=(415,415), scale=1/255, swapRB=True)

def ObjectDetector(image):
    classes, scores, boxes = model.detect(image, confidence_threshold, nms_threshold)
    for (classid, score, box) in zip(classes, scores, boxes):
        color=colors[int(classid)%len(colors)]
        label="%s:%f" % (class_names[classid],score)
        cv.rectangle(image, box, color, 2)
        cv.putText(frame,label,(box[0], box[1]-10), fonts, 0.5, color, 2)
camera=cv.VideoCapture(0)
counter=0
capture=False
number=0

while True:
    ret, frame = camera.read()
    original=frame.copy()
    ObjectDetector(frame)
    cv.imshow('original', original)
    print(capture == True and counter<10)

    if capture == True and counter<10:
        counter+=1
        cv.putText(frame,f"CaptureImg: {number}", (30,30), fonts, 0.6, pink,2)
    else:
        counter=0
    cv.imshow('frame',frame)
    key=cv.waitKey(1)

    if key==ord('c'):
        capture=True
        number+=1
        cv.imwrite(f'ReferenceImages/image{number}.png', original)
    if key == ord('q'):
        break

cv.destroyAllWindows()
