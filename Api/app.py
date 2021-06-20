from flask import Flask, request, jsonify
import numpy as np
import mimetypes
from PIL import Image
from preprocess import *
from model import *
app = Flask(__name__)

@app.route("/api/predict/video",methods=['POST'])
def predict_video():
    if request.method == 'POST':
        file = request.files['video']
        if(not is_file_type(file.filename,'video')):
            return handle_not_video_exception()
        np_images=split_video_to_np_images(file,frame_rate=30)
        feature_map=return_feature_map(np_images)
        if(feature_map==[]):
            return handle_no_hands_exception()
        np_right,np_left,np_two_hands,indexes=transform_data(feature_map)
        right_clf,left_clf,two_hands_clf=load_models()
        right_pred=right_clf.predict(np_right)
        left_pred=left_clf.predict(np_left)
        two_hands_pred=two_hands_clf.predict(np_two_hands)
        predictions=return_Predictions_array(right_pred,left_pred,two_hands_pred,indexes)
        return jsonify(results = predictions)

@app.route("/api/predict/image",methods=['POST'])
def predict_image():
    if request.method == 'POST':
        file = request.files['image']
        if(not is_file_type(file.filename,'image')):
            return handle_not_image_exception()
        np_image=np.array(Image.open(file))
        vect_img=[preprocess_image(np_image)]
        if vect_img==[None]:
            return handle_no_hands_exception()
        np_right,np_left,np_two_hands,_=transform_data(vect_img)
        if(np_right!=[]):
            clf=load_model('right_hand_logReg.pkl')
            np_data=np_right
        elif(np_left!=[]):
            clf=load_model('left_hand_randForest.pkl')
            np_data=np_left
        elif(np_two_hands!=[]):
            clf=load_model('two_hands_logReg.pkl')
            np_data=np_two_hands
        prediction=clf.predict(np_data).tolist()
        return jsonify(results = prediction)

@app.errorhandler(400)
def handle_no_hands_exception():
    return jsonify(error=400, message=str("There are no hands in the image"))

@app.errorhandler(415)
def handle_not_image_exception():
    return jsonify(error=415, message=str("The file is not an image"))

@app.errorhandler(415)
def handle_not_video_exception():
    return jsonify(error=415, message=str("The file is not a video"))


def is_file_type(filename,type):
    mimestart=type_of_file(filename)
    if(mimestart==type):
        return True
    return False
    
def type_of_file(filename):
    mimetypes.init()
    result = mimetypes.guess_type(filename)[0]
    if result != None:
        result = result.split('/')[0]
    return result