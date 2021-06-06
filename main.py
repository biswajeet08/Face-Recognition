import cv2
import json
import numpy

face_classifier = cv2.CascadeClassifier('resources/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

with open("dataset.json", "r+") as dataset:
    id_dict = json.load(dataset)

name = input('Name: ')
ID = input('ID: ')

if ID in id_dict.keys():
    print("OOPSss!!! ID already exist!!!")
    exit()
else:
    with open("dataset.json", "r+") as dataset:
        dict = {ID:name}
        data = json.load(dataset)
        data.update(dict)
        dataset.seek(0)
        json.dump(data, dataset, indent=4)
    count = 0
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count = count + 1
            dataset_path = "Dataset/" + name + "." + ID + "." + str(count) + ".jpg"
            cv2.imwrite(dataset_path, gray[y:y + h, x:x + w])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.waitKey(100)
        cv2.imshow("face", frame)
        cv2.waitKey(1)
        if count > 50:
            break

cap.release()
cv2.destroyAllWindows()
exit()
