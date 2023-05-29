import cv2
import os
dataset="AI"
name="Faces"
path=os.path.join(dataset,name)
if not os.path.isdir(path):
    os.makedirs(path)
width,height=130,100
count=1
alg="haarcascade_frontalface_default.xml"
haar_cascade=cv2.CascadeClassifier(alg)
cam=cv2.VideoCapture(0)
while count<31:
    print(count)
    _,img=cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face=haar_cascade.detectMultiScale(gray,1.3,4)
    for(x,y,w,h) in face:
        cv2.rectangle(img,(x,y),(x+w,y+h),(258,128,0),2)
        faceonly=gray[y:y+h,x:x+w]
        resize=cv2.resize(faceonly,(width,height))
        cv2.imwrite("%s/%s.jpg"%(path,count),resize)
        count+=1
    cv2.imshow("Face Detection",img)
    key=cv2.waitKey(10)
    if key == 27:
        break
print("Image Captured")
cam.release()
cv2.destroyAllWindows()
