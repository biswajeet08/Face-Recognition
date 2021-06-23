import cv2
import json
import numpy as np
import os

face_classifier = cv2.CascadeClassifier('resources/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
rec = cv2.face.LBPHFaceRecognizer_create()
rec.read("recognizer/trainedData.yml")

font = cv2.FONT_HERSHEY_SIMPLEX

with open("dataset.json", "r") as read_file:
    id_dict = json.load(read_file)

while (True):

    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, conf = rec.predict(gray[y:y + h, x:x + w])
        if conf < 500:
            conf = int(100 * (1 - conf / 300))
        if conf >= 75:
            if str(id) in id_dict.keys():
                id = id_dict.get(str(id))
                cv2.putText(img, id, (x, y + h), font, 2, (255, 0, 0), 3)
                cv2.imshow("Face", img)

        else:
            id = "Unknown"
            cv2.putText(img, id, (x, y + h), font, 2, (255, 0, 0), 3)
            cv2.imshow("Face", img)
    if (cv2.waitKey(1) == ord('q')):
        break
cap.release()
cv2.destroyAllWindows()
exit()
