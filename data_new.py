import numpy as np
import os
import cv2
import dlib

images1 = []
images2 = []
labels = []
folders_main = ['S1' , 'S2' , 'S3']

detector = dlib.get_frontal_face_detector()

def face_detector(image):

    faces = detector(image)
    face = faces[0]
    x1, y1 = face.left(), face.top()
    x2, y2 = face.right(), face.bottom()
    # cv2.rectangle(img1, (x1_1 - 20, y1_1 - 25), (x2_1 + 10, y2_1 + 10), (0, 0, 255), 2)
    roi = image[y1 - 20:y2 + 10, x1 - 20:x2 + 10]
    return roi

for sub_folder in folders_main:

    path = 'data/'+ sub_folder + '/'
    files = os.listdir(path)

    for i in range(len(files)):

        for j in range(len(files)):

            img1 = cv2.imread(path + files[i])
            img1 = face_detector(img1)
            img1 = cv2.resize(img1 , (128,128))
            img2 = cv2.imread(path + files[j])
            img2 = face_detector(img2)
            img2 = cv2.resize(img2 , (128,128))
            images1.append(img1)
            images2.append(img2)
            labels.append(1)

print(len(images2))

for sub_folder in folders_main:

    folders = ['S1' , 'S2' , 'S3']
    print(sub_folder)
    path = 'data/' + sub_folder + '/'
    folders.remove(sub_folder)
    rem_folder = folders

    print(rem_folder)

    files_similar = os.listdir(path)
    files_dissimilar1 = os.listdir('data/{}/'.format(rem_folder[0]))
    files_dissimilar2 = os.listdir('data/{}/'.format(rem_folder[1]))
    rem_images = files_dissimilar1 + files_dissimilar2

    print(rem_images)

    for i in range(len(files_similar)):
        k=0

        for j in range(len(rem_images)):

            img1 = cv2.imread(path+files_similar[i])
            img1 = face_detector(img1)
            img1 = cv2.resize(img1, (128, 128))
            img2 = cv2.imread('data/{}/'.format(rem_folder[k]) + rem_images[j])
            img2 = face_detector(img2)
            img2 = cv2.resize(img2, (128, 128))
            images1.append(img1)
            images2.append(img2)

            if j%5==4:
                k=k+1

            labels.append(0)

print(len(labels))

X1 = np.array(images1)
X2 = np.array(images2)
Y = np.array(labels)

np.save('numpy_files/X1.npy',X1)
np.save('numpy_files/X2.npy',X2)
np.save('numpy_files/Y.npy',Y)