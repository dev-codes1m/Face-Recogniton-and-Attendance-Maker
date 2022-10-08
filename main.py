import cv2
import numpy as np
import face_recognition

#Loading the image and converting them into RGB

imgElon = face_recognition.load_image_file('image basics/elon.jpg')
imgElon  = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)

imgTest = face_recognition.load_image_file('image basics/elon test.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

# Finding the face in our images
faceLoc = face_recognition.face_locations(imgElon)[0]
encodeELon = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,0),2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,0),2)


# Comparing these faces and also distance between there similarites

results = face_recognition.compare_faces([encodeELon],encodeTest)
faceDis = face_recognition.face_distance([encodeELon],encodeTest)
print(results,faceDis)
cv2.putText(imgTest,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
cv2.imshow('Elon Musk', imgElon)
cv2.imshow('Elon Musk Test',imgTest)
cv2.waitKey(0)