from flask import Flask, request
import numpy as np
import cv2

app = Flask(__name__)

@app.route("/",methods=['POST'])
def index():
    file = request.files['image'].read() ## byte file
    img=preprocess(file)
    return "hello world"


def preprocess(file):
    npimg = np.fromstring(file, np.uint8)
    img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)
    return img