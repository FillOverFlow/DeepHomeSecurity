import cv2 as cv2

file = '/home/davinci/AI/AnomalyDetection_CVPR18/input/Explosion008_x264.mp4'
cap = cv2.VideoCapture(file)

while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    else:
        break
