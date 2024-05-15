import cv2
from PIL import Image
import numpy as np
import os

if not os.path.exists('frames'):
    os.makedirs('frames')

def VidClip(video_path1,video_path2, output_path):
    cap1 = cv2.VideoCapture(video_path1)
    cap2 = cv2.VideoCapture(video_path2)

    width, height = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))
    if not cap1.isOpened() or not cap2.isOpened():
        print("Error: Could not open video file.")
        return
    
    count = 0
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            break
        frame1c=frame1.copy()
        frame2c=frame2.copy()  

        start_x1, start_y1 = 0, 0
        end_x1, end_y1 = 960, 1080
        frame1c[start_y1:end_y1, start_x1:end_x1] = (0, 0, 0)

        start_x2, start_y2 = 1032,0
        end_x2, end_y2 = 1920, 1920
        frame2c[start_y2:end_y2, start_x2:end_x2] = (0, 0, 0)

        image1 = Image.fromarray((frame1c * 255).astype(np.uint8))
        image2 = Image.fromarray((frame2c * 255).astype(np.uint8))

        rgba1 = image1.convert("RGBA")
        rgba2 = image2.convert("RGBA")

        datas1 = rgba1.getdata()
        new_data1 = []
        for item in datas1:
            if item[0] == 0 and item[1] == 0 and item[2] == 0:
                new_data1.append((255, 255, 255, 0))
        else:
            new_data1.append(item)

        datas2 = rgba1.getdata()
        new_data2 = []
        for item in datas2:
            if item[0] == 0 and item[1] == 0 and item[2] == 0:
                new_data2.append((255, 255, 255, 0))
        else:
            new_data2.append(item)
        rgba1.putdata(new_data1)
        rgba2.putdata(new_data1)

        frame1 = cv2.cvtColor(np.array(rgba1), cv2.COLOR_RGBA2BGR)
        frame2 = cv2.cvtColor(np.array(rgba2), cv2.COLOR_RGBA2BGR)

    # Save the frames.
        cv2.imwrite(f'frames/frame1_{count}.png', frame1)
        cv2.imwrite(f'frames/frame2_{count}.png', frame2)

        count += 1
        images=[rgba1,rgba2]
        total_width = sum(image.width for image in images)
        max_height = max(image.height for image in images)
        new_im = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for im in images:
            new_im.paste(im, (x_offset, 0))
            x_offset += im.width
        frame = cv2.cvtColor(np.array(new_im), cv2.COLOR_RGB2BGR)
        out.write(frame)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap1.release()
    cap2.release()
    out.release()

video_path1 = "WIN_20240506_00_19_15_Pro.mp4"
video_path2 = "WIN_20240506_00_19_38_Pro.mp4"
output_path = "output_combined_video.mp4"
VidClip(video_path1, video_path2, output_path)