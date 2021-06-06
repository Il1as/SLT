from flask import Flask, request
import numpy as np
import cv2
import mediapipe as mp
import os

app = Flask(__name__)

@app.route("/api/video/predict",methods=['POST'])
def predict():
    file = request.files['video']
    if file.filename != '':
        file.save(file.filename)
    print("video written")
    images=split_video_to_images(file) 
    print(images)
    #vect_img=preprocess(file)
    return "hello world"

@app.route("/api/load_model",methods=['GET'])
def load():
    load_model()


def split_video_to_images(file):
    print("split video to images")
    result=[]
    vidcap = cv2.VideoCapture(file.filename)
    success,image = vidcap.read()
    count = 0
    while success:
        result.append(image)
        success,image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1
    vidcap.release()
    os.remove(file.filename)
    return result

def load_model():
    pass

def preprocess(file):
    npimg = np.fromstring(file, np.uint8)
    img = cv2.imdecode(npimg, cv2.COLOR_BGR2RGB)
    img_vect=img_to_vector(img)
    return img_vect



def img_to_vector(image):
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    row=[]
    with mp_holistic.Holistic(static_image_mode=True,model_complexity=2) as holistic:
        # Make Detections
        results = holistic.process(image)
        print(results.pose_landmarks)
        print(results.left_hand_landmarks)      
        print(results.right_hand_landmarks)  
        # Export coordinates
        # If some landmarks are not visible they are replaced with Nan in the dataset
        
        try:
            
            #if no hand is in the camera, don't insert the row    
            if(results.left_hand_landmarks==None and results.right_hand_landmarks==None):
                raise Exception('There is no hand in the picture')    
                
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
            
            # Concate rows
            row = pose_row+right_row+left_row
            return row

        except Exception as e:
            print("Error : ",e)
