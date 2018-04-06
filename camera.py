import cv2

cap = cv2.VideoCapture(0)
# cascPath = "haarcascade_frontalface_default.xml"
cascPath = "haarcascade_eye.xml"
# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)
while(1):
    # get a frame
    ret, frame = cap.read()
    # show a frame
    # cv2.imshow("capture", frame)
    # 显示人脸
    image =frame
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
        #flags = cv2.CV_HAAR_SCALE_IMAGE
    )
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Faces found", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows() 