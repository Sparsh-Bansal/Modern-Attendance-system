from tensorflow.python.keras.models import load_model
import numpy as np
import os
import cv2
import dlib
import argparse

parser = argparse.ArgumentParser()
# parser.add_argument('--image' , required=True)
args = parser.parse_args()

detector = dlib.get_frontal_face_detector()

loaded_model = load_model('model_num.hdf5')

def face_detector(image):
    faces = detector(image)
    face = faces[0]
    x1, y1 = face.left(), face.top()
    x2, y2 = face.right(), face.bottom()
    # cv2.rectangle(img1, (x1_1 - 20, y1_1 - 25), (x2_1 + 10, y2_1 + 10), (0, 0, 255), 2)
    roi = image[y1 - 20:y2 + 10, x1 - 20:x2 + 10]
    return roi


def process_image(sub_folder):
    images = []
    folder = 'data/'+sub_folder+'/'
    files = os.listdir(folder)
    print(files)
    for file in files:
        img = cv2.imread(folder + file)
        img = face_detector(img)
        img = cv2.resize(img , (128 , 128))
        images.append(img)
    print(len(images))
    images = np.array(images)
    return images

aamir = process_image('S1')
salmaan = process_image('S2')
shahrukh = process_image('S3')

input_image = cv2.imread('shah6.jfif')
input_image = face_detector(input_image)
input_image = cv2.resize(input_image , (128 ,128))
print(len(input_image))
input_image = input_image.reshape( ( 1, 128,128,3 ) ).astype( np.float32 )
print(input_image.shape)

dic = {0: 'Aamir Khan' , 1 : 'Salmaan Khan' , 2 : 'Shahrukh Khan'}

scores = []
label = []
for im in aamir:
    im = im.reshape((1, 128, 128, 3)).astype(np.float32)
    score = loaded_model.predict([im , input_image])[0]
    scores.append(score)
    label.append(0)

for im in salmaan:
    im = im.reshape((1, 128, 128, 3)).astype(np.float32)
    score = loaded_model.predict([im , input_image])[0]
    scores.append(score)
    label.append(1)

for im in shahrukh:
    im = im.reshape((1, 128, 128, 3)).astype(np.float32)
    score = loaded_model.predict([im , input_image])[0]
    scores.append(score)
    label.append(2)
print(scores)
index = np.argmax(scores)
_label_ = label[index]

print(dic[_label_])



