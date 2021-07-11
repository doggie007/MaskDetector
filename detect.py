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

    faces = []
    locations = []
    predictions = []

    for i in range(0,detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence > 0.4:
            box = detections[0,0,i,3:7] * np.array([w,h,w,h])
            startX, startY, endX, endY = box.astype("int")
            startX, startY = max(0,startX),max(0,startY)
            endX, endY = min(w-1, endX), min(h-1, endY)

            face = frame[startY:endY, startX:endX]
            try:
                face = cv2.resize(face, (128,128))
                face = face[...,::-1].astype(np.float32) / 255.0
                face = np.reshape(face, (1,128,128,3))
                faces.append(face)
                locations.append((startX, startY, endX, endY))
                prediction = mask_nn.predict(face)
                prediction = np.argmax(prediction, axis=-1)[0]
                predictions.append(prediction)
            except:
                pass
        
    return (locations,predictions)

mapping = {0:"mask worn incorrectly",1:"with mask",2:"without mask"}

prototxtPath = "face_detector/deploy.prototxt"
weightsPath = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"

face_nn = cv2.dnn.readNet(prototxtPath,weightsPath)

mask_nn = load_model("model_nn")

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
        cv2.putText(frame, mapping[prediction],(startX, startY - 20),cv2.FONT_HERSHEY_COMPLEX, 0.55, color, 2)
    cv2.imshow("LIVE", frame)
    key = cv2.waitKey(10) 
    if key == ord('q'):
        break

cv2.destroyAllWindows()
vs.stop()