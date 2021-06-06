import cv2
import numpy as np
import  os
from os import listdir
from os.path import isfile, join
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()
data_path = 'Dataset'

def getImagesWithName(data_path):
    images_path = [join(data_path,f) for f in listdir(data_path) if f.endswith('.jpg')]
    print(images_path)
    faces = []
    IDs = []
    for imagepath in images_path:
        faceImg = Image.open(imagepath).convert('L')
        faceNp = np.array(faceImg,"uint8")
        ID = os.path.split(imagepath)[-1].split('.')[1]
        faces.append(faceNp)
        IDs.append(ID)
        cv2.imshow('Training',faceNp)
        cv2.waitKey(10)
    return IDs, faces

IDs, faces = getImagesWithName(data_path)
IDs = np.asarray(IDs,dtype=np.int32)
recognizer.train(faces, np.asarray(IDs))
recognizer.save("recognizer/trainedData.yml")
cv2.destroyAllWindows()




# data_path = 'Dataset/'
# only_files = [f for f in listdir(data_path) if isfile(join(data_path,f)) and f.endswith('.jpg')]
#
# Training_data, lables = [], []
#
# for i, files in enumerate(only_files):
#     image_path = data_path + only_files[i]
#     images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     Training_data.append(np.asarray(images,dtype=np.uint8))
#     lables.append(i)
#
# lables = np.asarray(lables,dtype=np.int32)
#
# model = cv2.face.LBPHFaceRecognizer_create()
# model.train(np.asarray(Training_data), np.asarray(lables))
