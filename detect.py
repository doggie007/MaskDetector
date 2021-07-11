"""

TEST WITH IMAGE

#stop tensorflow output / warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np 
from keras.models import load_model

model = load_model("model_nn")

img = cv2.imread("testing/maskIncorrect.jpg")


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Convert into grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.1, 4)
print(faces)


mapping = {0:"mask weared incorrectly",1:"with mask",2:"without mask"}


print(mapping)
#Draw rectangle around the faces
for (x, y, w, h) in faces:
    face = img[y:y+h,x:x+w]
    face = cv2.resize(face,(128,128))
    face = face[...,::-1].astype(np.float32) / 255.0
    face = np.reshape(face,(1,128,128,3))
    result = model.predict(face)

    label = np.argmax(result,axis=-1)[0]
    if label == 0:
        color = (0,165,255)
    elif label == 1:
        color = (255,0,0)
    else:
        color = (0,0,255)
    cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
    cv2.putText(img,mapping[label],(x,y-20),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)
    
cv2.imshow('img', img)
cv2.waitKey()
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np 
from keras.models import load_model

model = load_model("model_nn")
# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

mapping = {0:"mask weared incorrectly",1:"with mask",2:"without mask"}

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame,1,1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        face = frame[y:y+h,x:x+w]
        test = face
        test = cv2.resize(test,(128,128))
        test = test[...,::-1].astype(np.float32) / 255.0
        test = np.reshape(test,(1,128,128,3))
        result = model.predict(test)

        confidence = np.max(result)
        if confidence < 0.4:
            continue
        label = np.argmax(result,axis=-1)[0]
        if label == 0:
            color = (0,165,255)
        elif label == 1:
            color = (255,0,0)
        elif label == 2:
            color = (0,0,255)
        cv2.rectangle(frame, (x-30, y-30), (x+w+30, y+h+30), color, 2)
        cv2.putText(frame,mapping[label],(x,y-20),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)

    cv2.imshow("LIVE", frame)
    key = cv2.waitKey(10)
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


# for (x, y, w, h) in faces:
#     face = img[y:y+h,x:x+w]
#     resized_face = cv2.resize(face,(128,128))
#     normalized_face = resized_face/255.0
#     reshaped_face = np.reshape(normalized_face,(1,128,128,3))
#     result = model.predict(reshaped_face)

#     label = np.argmax(result,axis=-1)[0]
#     if label == 0:
#         color = (0,165,255)
#     elif label == 1:
#         color = (255,0,0)
#     else:
#         color = (0,0,255)
#     cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
#     cv2.putText(img,mapping[label],(x,y-20),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)
    
    
    



# # Display the output

# cv2.imshow('img', img)
# cv2.waitKey()


"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os



def detect_mask(frame, face_nn, mask_nn):
    (h,w) = frame.shape[:2]
    blob  = cv2.dnn.blobFromImage(frame,1.0,(224,224), (104.0,177.0,123.0))
    face_nn.setInput(blob)
    detections = face_nn.forward()
    print(detections.shape)

    faces = []
    locations = []
    predictions = []

    for i in range(0,detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence > 0.2:
            box = detections[0,0,i,3:7] * np.array([w,h,w,h])
            startX, startY, endX, endY = box.astype("int")
            startX, startY = max(0,startX),max(0,startY)
            endX, endY = min(w-1, endX), min(h-1, endY)

            face = frame[startY:endY, startX:endX]
            face = cv2.resize(face, (128,128))
            face = face[...,::-1].astype(np.float32) / 255.0
            face = np.reshape(face, (1,128,128,3))
            faces.append(face)
            locations.append((startX, startY, endX, endY))
            prediction = mask_nn.predict(face)
            prediction = np.argmax(prediction, axis=-1)[0]
            predictions.append(prediction)
        
    return (locations,predictions)

mapping = {0:"mask weared incorrectly",1:"with mask",2:"without mask"}

prototxtPath = "face_detector/deploy.prototxt"
weightsPath = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"

face_nn = cv2.dnn.readNet(prototxtPath,weightsPath)

mask_nn = load_model("second")

print("Starting video stream")

vs = VideoStream(src=0).start()

while True:
    frame = vs.read()
    # frame = imutils.resize(frame, width=)
    locations, predictions = detect_mask(frame, face_nn, mask_nn)
    for box, prediction in zip(locations, predictions):
        startX, startY, endX, endY = box
        if prediction == 0:
            color = (0,165,255)
        elif prediction == 1:
            color = (255,0,0)
        else:
            color = (0,0,255)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        cv2.putText(frame, mapping[prediction],(startX, startY - 20),cv2.FONT_HERSHEY_COMPLEX, 0.45, color, 2)
    cv2.imshow("LIVE", frame)
    key = cv2.waitKey(10) 
    if key == ord('q'):
        break

cv2.destroyAllWindows()
vs.stop()
"""