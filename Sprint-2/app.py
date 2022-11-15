from flask import Flask, render_template, Response, url_for
import cv2
import numpy as np
import os
from keras.models import load_model
import tensorflow as tf
from gtts import gTTS
global graph
global writer
from skimage.transform import resize

graph = tf.compat.v1.get_default_graph()
writer = None

model = load_model('asl_model.h5')

vals = ['A','B','C','D','E','F','G','H','I']

app = Flask(__name__)
print("[INFO] accessing video stream...")

pred = ""
def gen():
    vid = cv2.VideoCapture(0)
    vid.set(cv2.CAP_PROP_AUTOFOCUS,0)
    vid.set(3,1280)
    vid.set(4,720)
    while (True):
        ret, frame = vid.read()
        #detect(frame)
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.flip(gray_image,1)
        prediction = detect(gray_image)
        cv2.putText(frame,'The Predicted Alphabet is: '+str(prediction),(100,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
        _, image_binary = cv2.imencode('.jpg',gray_image)
        binary_data = image_binary.tobytes()
        data_frame = b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + binary_data + b'\r\n\r\n'

        
        _, col_img = cv2.imencode('.jpg',frame)
        col_img = cv2.flip(col_img,1)
        col_binary = col_img.tobytes()
        col_data_frame = b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + col_binary + b'\r\n\r\n'
        yield col_data_frame


def detect(frame):
    img = resize(frame,(64,64,3))
    img = np.expand_dims(img,axis=0)
    if (np.max(img)>1):
        img = img/255.0
    model = load_model('asl_model.h5')
    prediction = model.predict(img)
    prediction = np.argmax(prediction,axis=1)
    pred = vals[prediction[0]]
    return pred
    
    
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype = 'multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug = True)
