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

# id_dict = {1: 'Vicky', 2: 'Surili', 3: 'Sam', 4: 'Dika', 5: 'Sangram', 6: 'Gyana'}

while (True):

    ret, img = cap.read();
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, conf = rec.predict(gray[y:y + h, x:x + w])
        if conf < 500:
            conf = int(100 * (1 - conf / 300))
        if conf >= 55:
            if str(id) in id_dict.keys():
                id = id_dict.get(str(id))
                cv2.putText(img, id, (x, y + h), font, 2, (255, 0, 0), 3)
                cv2.imshow("Face", img)

        # if id == 1:
        #     id = "Vicky"
        #     cv2.putText(img, id, (x, y + h), font, 2, (255, 0, 0), 3)
        #     cv2.imshow("Face", img)
        # elif id == 2:
        #     id = "Surili"
        #     cv2.putText(img, id, (x, y + h), font, 2, (255, 0, 0), 3)
        #     cv2.imshow("Face", img)
        # elif id == 3:
        #     id = "Sam"
        #     cv2.putText(img, id, (x, y + h), font, 2, (255, 0, 0), 3)
        #     cv2.imshow("Face", img)
        # elif id == 4:
        #     id = "Dika"
        #     cv2.putText(img, id, (x, y + h), font, 2, (255, 0, 0), 3)
        #     cv2.imshow("Face", img)
        # elif id == 5:
        #     id = "Sangram"
        #     cv2.putText(img, id, (x, y + h), font, 2, (255, 0, 0), 3)
        #     cv2.imshow("Face", img)
        # elif id == 6:
        #     id = "Gyana"
        #     cv2.putText(img, id, (x, y + h), font, 2, (255, 0, 0), 3)
        #     cv2.imshow("Face", img)
        else:
            id = "Unknown"
            cv2.putText(img, id, (x, y + h), font, 2, (255, 0, 0), 3)
            cv2.imshow("Face", img)
    if (cv2.waitKey(1) == ord('q')):
        break
cap.release()
cv2.destroyAllWindows()
exit()
