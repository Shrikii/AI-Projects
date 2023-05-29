import cv2
import time
import imutils
cam=cv2.VideoCapture(0)
time.sleep(1)
first_frame=None
area=500
while True:
    text="No Object Detected"
    _,img=cam.read()
    img=imutils.resize(img,width=500)
    gray_Img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gauss_Img=cv2.GaussianBlur(gray_Img,(21,21),0)
    if first_frame is None:
        first_frame=gauss_Img
        continue
    img_diff=cv2.absdiff(first_frame,gauss_Img)
    thresh_Img=cv2.threshold(img_diff,25,255,cv2.THRESH_BINARY)[1]
    thresh_Img=cv2.dilate(thresh_Img,None,iterations=2)
    cnts=cv2.findContours(thresh_Img.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts=imutils.grab_contours(cnts)
    for c in cnts:
            if cv2.contourArea(c) < area:
                continue
            (x,y,w,h)=cv2.boundingRect(c)
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            text="Moving object detected"
    print(text)
    cv2.putText(img,text,(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
    cv2.imshow("Default Camera",img)
    key=cv2.waitKey(1) & 0xFF
    if key==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
