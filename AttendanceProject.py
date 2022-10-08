import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'Image Attendance'
images = []
classNames = []
myList = os.listdir(path)
print(myList)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

# Face encoding Process
def findencodings(images):
    encodeList = []
    for img in images:
        img  = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)
        encodeList.append(encode)
    return encodeList[0]
# Make Attendance Function
def markAttendacne(name):
    with open('attendance.csv','r+') as f:
        myDataList = f.readline()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H,%M,%S')
            f.writelines(f'\n{name},{dtString}')
        print(myDataList)
encodeListKnown = findencodings(images)
print('Encoding Complete')

# Matching of images which we dont have so we are using our webcam to initialise the image
cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS,faceCurFrame)

    for encodeface,faceLoc in zip(encodesCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeface)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeface)
        matchIndex = np.argmin(faceDis)
        #print(faceDis)
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            #print(name)
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendacne(name)

    cv2.imshow('Webcam',img)
    cv2.waitKey(1)


# faceLoc = face_recognition.face_locations(imgElon)[0]
# encodeELon = face_recognition.face_encodings(imgElon)[0]
# cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,0),2)
#
# faceLocTest = face_recognition.face_locations(imgTest)[0]
# encodeTest = face_recognition.face_encodings(imgTest)[0]
# cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,0),2)
#
#
# # Comparing these faces and also distance between there similarites
#
# results = face_recognition.compare_faces([encodeELon],encodeTest)
# faceDis = face_recognition.face_distance([encodeELon],encodeTest)


#Loading the image and converting them into RGB

# imgElon = face_recognition.load_image_file('image basics/elon.jpg')
# imgElon  = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)
#
# imgTest = face_recognition.load_image_file('image basics/elon test.jpg')
# imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)