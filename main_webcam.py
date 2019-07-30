from tensorflow.python.keras.models import load_model
import numpy as np
import os
import cv2
import dlib

detector = dlib.get_frontal_face_detector()

loaded_model = load_model('model_num.hdf5')

def face_detector(image):
    faces = detector(image)
    print(faces)
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
    # print(files)
    for file in files:
        img = cv2.imread(folder + file)
        img = face_detector(img)
        img = cv2.resize(img , (128 , 128))
        images.append(img)
    # print(len(images))
    images = np.array(images)
    return images

def predict():

    dic = {0: 'Aamir Khan' , 1 : 'Salmaan Khan' , 2 : 'Shahrukh Khan'}

    input_image = cv2.imread('input_webcam.jpg')
    input_image = face_detector(input_image)
    input_image = cv2.resize(input_image, (128, 128))
    print(len(input_image))
    input_image = input_image.reshape((1, 128, 128, 3)).astype(np.float32)
    print(input_image.shape)
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

    index = np.argmax(scores)
    _label_ = label[index]

    print(dic[_label_])
    return dic[_label_]

aamir = process_image('S1')
salmaan = process_image('S2')
shahrukh = process_image('S3')

cam = cv2.VideoCapture(0)

text = 'Face Detected.Press key "c" \nfor the attendance'
while(True):
    ret , input_image = cam.read()
    prediction_image = input_image
    # input_image = np.flip(input_image , axis=1)
    faces  = detector(input_image)
    for points in faces:
        tlx ,tly = points.left() , points.top()
        brx , bry = points.right() , points.bottom()
        cv2.rectangle(input_image , (tlx ,tly) , (brx,bry) , (0,0,255),2)

    if len(faces)==0:
        cv2.putText(input_image, 'No Face Detected', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
    if len(faces)>1:
        cv2.putText(input_image, 'Only one face at a time', (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    if len(faces) ==1:
        if cv2.waitKey(1) & 0xFF == ord('c'):
            cv2.imwrite('input_webcam.jpg' , prediction_image)
            pred_text = predict()
            text = 'Hey ' + pred_text + ' Your Attendace is complete'

        cv2.putText(input_image, text, (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)

    cv2.imshow('Frame' , input_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()


