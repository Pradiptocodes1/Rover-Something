# from vidgear.gears import VideoGear
# import cv2
# import time
# stream1 = VideoGear(source=0, logging=True).start() 
# stream2 = VideoGear(source=1, logging=True).start() 
# while True: 
#     frame1 = stream1.read()
#     frame2 = stream2.read()
#     if frame1 is None or frame2 is None:
#         break
#     cv2.imshow("Output Frame1", frame1)
#     cv2.imshow("Output Frame2", frame2)
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord("q"):
#         break

#     if key == ord("w"):
#         cv2.imwrite("Image-1.jpg", frame1)
#         cv2.imwrite("Image-2.jpg", frame2)

# cv2.destroyAllWindows()
# stream1.stop()

from vidgear.gears import VideoGear
import cv2
import numpy as np
import time

stream1 = VideoGear(source=0, logging=True).start() 
stream2 = VideoGear(source=1, logging=True).start() 

while True: 
    frame1 = stream1.read()
    frame2 = stream2.read()
    print("Frame1 shape:", frame1.shape)
    print("Frame2 shape:", frame2.shape)

    if frame1 is None or frame2 is None:
        break

    start_x, start_y = 0, 0
    end_x, end_y = 511, 480
    crop1 = frame1[start_y:end_y, start_x:end_x]

    start_x1, start_y1 = 129, 0
    end_x1, end_y1 = 640, 480
    crop2 = frame2[start_y1:end_y1, start_x1:end_x1]

    # crop1 = cv2.resize(crop1, (1080, 1920))
    # crop2 = cv2.resize(crop2, (1080, 1920))

    joined_frame = np.concatenate((frame1, frame2), axis=1)
    joined_frame_c = np.concatenate((crop1, crop2), axis=1)

    # cv2.imshow("Output Frame1", frame2)
    # cv2.imshow("Output Frame2", frame1)
    # cv2.imshow("Cropped Frame 1", frame2)
    # cv2.imshow("Cropped Frame 2", frame1)
    cv2.imshow('Joined Frame', joined_frame)
    cv2.imshow('Joined Frame(Cropped)', joined_frame_c)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

    if key == ord("w"):
        cv2.imwrite("Image-1.jpg", frame1)
        cv2.imwrite("Image-2.jpg", frame2)

cv2.destroyAllWindows()
stream1.stop()
stream2.stop()
