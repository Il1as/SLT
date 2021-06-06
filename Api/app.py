from flask import Flask, request
import numpy as np
import cv2
import mediapipe as mp
import os

app = Flask(__name__)

@app.route("/api/video/predict",methods=['POST'])
def predict_video():
    file = request.files['video']
    save_file(file)
    np_images=split_video_to_np_images(file,frame_rate=30) 
    delete_file(file)
    feature_map=return_feature_map(np_images)
    return "hello world"

@app.route("/api/image/predict",methods=['POST'])
def predict_image():
    file = request.files['image']
    np_image = np.fromstring(file, np.uint8)
    vect_img=preprocess(np_image)
    return "hello world"

def save_file(file):
    if file.filename != '':
        file.save(file.filename)

def delete_file(file):
    os.remove(file.filename)

def return_feature_map(np_images):
    result=[]
    for np_image in np_images:
        preprocessed_img=preprocess(np_image)
        if preprocessed_img!=None:
            result.append(preprocessed_img)
    return result

def split_video_to_np_images(file,frame_rate=30):
    result=[]
    vidcap = cv2.VideoCapture(file.filename)
    success,image = vidcap.read()
    count = 0
    while success:
        if(count % frame_rate==0):
            result.append(image)
        success,image = vidcap.read()
        count += 1
    vidcap.release()
    return result

def load_model():
    pass

def preprocess(np_image):
    img = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
    return img_to_vector(img)


def img_to_vector(image):
    mp_holistic = mp.solutions.holistic
    result=[]
    with mp_holistic.Holistic(static_image_mode=True,model_complexity=2) as holistic:
        
        # Make Detections
        results = holistic.process(image)
        
        try:
            
            #if no hand is in the camera, don't insert the row    
            if(results.left_hand_landmarks==None and results.right_hand_landmarks==None):
                raise Exception('There is no hand in the picture')    

            # Extract pose landmarks
            if(results.pose_landmarks==None):
                pose_row=list(np.array(([np.NaN])*132).flatten())
            else:
                pose = results.pose_landmarks.landmark
                pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
                
            # Extract right hand landmarks
            if(results.right_hand_landmarks==None):
                right_row=list(np.array(([np.NaN])*84).flatten())
            else:
                right = results.right_hand_landmarks.landmark
                right_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in right]).flatten())
            
            # Extract left hand landmarks
            if(results.left_hand_landmarks==None):
                left_row=list(np.array(([np.NaN])*84).flatten())
            else:
                left = results.left_hand_landmarks.landmark
                left_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in left]).flatten())
            
            # Concatenate rows
            result = pose_row+right_row+left_row

            return result

        except Exception as e:
            pass
