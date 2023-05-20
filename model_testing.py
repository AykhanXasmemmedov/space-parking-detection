
import cv2
import cardetect
import anno_txt_read
import os
import time
import numpy as np


def intersection_over_union(real,predict):
    x0=max(real[0],predict[0])
    y0=max(real[1],predict[1])
    x1=min(real[2],predict[2])
    y1=min(real[3],predict[3])
    
    interArea=max(0,x1-x0+1)*max(0,y1-y0+1)
    
    realArea=(real[2]-real[0]+1)*(real[3]-real[1]+1)
    predictArea=(predict[2]-predict[0]+1)*(predict[3]-predict[1]+1)
    
    iou=interArea/float(realArea+predictArea-interArea)
    
    return iou


main_path=os.getcwd()

video_path=os.path.join(main_path,'parking.mp4')

video=cv2.VideoCapture(video_path)

_,frame=video.read()
h,w=frame.shape[:2]
box_coordinate=anno_txt_read.annotation_txt_read('image.txt',h,w)

prev_time = 0
new_time = 0
while video.isOpened():
    
    ret,frame=video.read()
    
    if not ret:
        break

    car_num,coordinates=cardetect.detect_car(frame)
    
    collapsed_space=1
    
    for park_coor in box_coordinate: 
        cv2.rectangle(frame,(park_coor[0],park_coor[1]),(park_coor[2],park_coor[3]),(0,255,0),3)

        for predict_coor in coordinates:
            roi=int((intersection_over_union(park_coor, predict_coor))*100)
            
            if roi>40:
                frame=cv2.rectangle(frame,(park_coor[0],park_coor[1]),(park_coor[2],park_coor[3]),(0,0,255),3)
                cv2.rectangle(frame,(park_coor[0],park_coor[1]),(park_coor[2],park_coor[3]),(0,0,255),2)
                
                collapsed_space=collapsed_space+1
            
    empty_space=len(box_coordinate)-collapsed_space
    cv2.putText(frame,f'exist empty space:{str(empty_space)}',(20,185),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.putText(frame,f'exist collapsed space:{str(collapsed_space)}',(20,215),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

    
    new_time = time.time()
    fps = int(1/(new_time-prev_time))
    cv2.putText(frame, f'FPS:{str(fps)}', (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    prev_time = new_time
    
    frame=cv2.resize(frame,(0,0),fx=1,fy=.7)
    cv2.imshow('frame', frame)
    key = cv2.waitKey(7)
    if key == ord('q'):
        break
    elif key==ord('w'):
        cv2.waitKey(0)
        

cv2.destroyAllWindows()
video.release()
