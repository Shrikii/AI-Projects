import cv2
import imutils
red_lower=(36, 100, 50)
red_upper=(90, 255, 255)
cam=cv2.VideoCapture(0)
while True:
    _,frame=cam.read()
    frame=imutils.resize(frame,width=600)
    gauss_blurr=cv2.GaussianBlur(frame,(11,11),0)
    hsv=cv2.cvtColor(gauss_blurr,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv,red_lower,red_upper)
    mask=cv2.erode(mask,None,iterations=2)
    mask=cv2.dilate(mask,None,iterations=2)
    cnts=cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
    center=None
    if len(cnts)>0:
        c=max(cnts,key=cv2.contourArea)
        ((x,y),radius)=cv2.minEnclosingCircle(c)
        M=cv2.moments(c)
        center=(int(M["m10"] / M["m00"])),(int(M["m01"] / M["m00"]))
        if radius>10:
            cv2.circle(frame,(int(x),int(y)),int(radius),(0,255,255),2)
            cv2.circle(frame,center,5,(0,255,255),-1)
            if radius>250:
                print("STOP")
            else:
                if (center[0]<150):
                    print("Left")
                elif(center[0]>450):
                    print("Right")
                elif(radius<250):
                    print("Front")
                else:
                    print("STOP")
    cv2.imshow("Object Tracking",frame)
    key=cv2.waitKey(2) & 0xFF
    if key==27:
        break
    
cam.release()
cv2.destroyAllWindows()

