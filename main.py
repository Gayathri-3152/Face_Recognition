import cv2
import numpy as np
import face_recognition

imgDhoni=face_recognition.load_image_file('Face_to_be_recognized/Dhoni.jpg')
imgDhoni=cv2.cvtColor(imgDhoni,cv2.COLOR_BGR2RGB)

imgtest=face_recognition.load_image_file('Face_to_be_recognized/Dhoni_test.png')
imgtest=cv2.cvtColor(imgtest,cv2.COLOR_BGR2RGB)

faceloc = face_recognition.face_locations(imgDhoni)[0]
encodeDhoni = face_recognition.face_encodings(imgDhoni)[0]
cv2.rectangle(imgDhoni,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255),2)

facelocTest = face_recognition.face_locations(imgtest)[0]
encodeTest = face_recognition.face_encodings(imgtest)[0]
cv2.rectangle(imgtest,(facelocTest[3],facelocTest[0]),(facelocTest[1],facelocTest[2]),(255,0,255),2)

results = face_recognition.compare_faces([encodeDhoni],encodeTest)
faceDis = face_recognition.face_distance([encodeDhoni],encodeTest)
print(results,faceDis)
cv2.putText(imgtest,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_TRIPLEX,0.75,(0,0,255),2)

cv2.imshow('Dhoni',imgDhoni)
cv2.imshow('Dhoni Test',imgtest)
cv2.waitKey(0)