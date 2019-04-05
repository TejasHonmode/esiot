import cv2
from scipy.spatial import distance as dist
import numpy as np
import h5py
import pickle
import keras
from keras.models import *

def get_id(photo):
    min_dist = 200
    thresh = 0.69
    
    photo = cv2.resize(photo, (96,96))
    photo = np.reshape(photo, (1,96,96,3))
    
    for name, embed in encodings.items():
        euclidean_distance = np.linalg.norm(encodings[name] - model.predict(photo))
        if euclidean_distance < min_dist:
            min_dist = euclidean_distance
            key = name
            
    if min_dist >= thresh:
        print('Euclidean distance : {}'.format(min_dist))
        return "Unknown"
    else:
        print('Euclidean distance : {}'.format(min_dist))
#         cv2.imwrite('/home/abc/xyx/Unknown'+str(i)+'.jpg',photo)
        return key

def get_face(path):
    face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    photo = cv2.imread(path, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(photo, 1.3, 5)
    for (x,y,w,h) in faces:
        cv.rectangle(photo,(x,y),(x+w,y+h),(255,0,0),2)
        roi = photo[y:y+h, x:x+w]
        
        name = get_face(roi)
    return name

model = load_model('model_face.h5', compile=False)

encodings = pickle.load(open('encodings.pkl','rb'))

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

#cap = cv2.VideoCapture(0)

"""while True: 
    ret, frame = cap.read() 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    #rects = detector(gray, 0)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        frame_roi = frame[y:y+h, x:x+w]
        pred_roi = cv2.resize(frame_roi, (96,96))
        #pred_roi = np.array(pred_roi)
        pred_roi = np.reshape(pred_roi,(1,96,96,3))
        name = get_face(pred_roi)
        
        cv2.putText(frame, "Face #{}".format(name), (x - 10, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 3)

    frame = cv2.resize(frame, (900,900))
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
"""
cap.release() 
cv2.destroyAllWindows()
