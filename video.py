import cv2
import numpy as np
import sys

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture(sys.argv[1])

# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
sift = cv2.xfeatures2d.SIFT_create()
# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    img_canny = cv2.Canny(gray, 100 , 150)  


    # # kp 是所有128特征描述子的集合
    # kp = sift.detect(gray, None)
    # # 找到后可以计算关键点的描述符
    # Kp, res = sift.compute(gray, kp)
    # # 还可以用下面的函数直接检测并返回特征描述符
    # kp2, res1 = sift.detectAndCompute(gray, None)
    # frame = cv2.drawKeypoints(frame, kp, frame,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow("sift", img_canny)


    # Display the resulting frame
    #cv2.imshow('Frame',frame)

    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

  # Break the loop
  else: 
    break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()