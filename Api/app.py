from flask import Flask, request, jsonify
import numpy as np
import cv2
import mediapipe as mp
import os
import pickle
import pandas as pd
import mimetypes

app = Flask(__name__)

@app.route("/api/predict/video",methods=['POST'])
def predict_video():
    file = request.files['video']
    save_file(file)
    if(not is_file_type(file,'video')):
        return handle_not_video_exception()
    np_images=split_video_to_np_images(file,frame_rate=30)
    delete_file(file)
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
    file = request.files['image']
    save_file(file)
    if(not is_file_type(file,'image')):
        return handle_not_image_exception()
    np_image=cv2.imread(file.filename)
    delete_file(file)
    vect_img=[preprocess_image(np_image)]
    if vect_img==[None]:
        return handle_no_hands_exception()
    np_right,np_left,np_two_hands,_=transform_data(vect_img)
    if(np_right!=[]):
        clf=load_model('right_hand_lr.pkl')
        np_data=np_right
    elif(np_left!=[]):
        clf=load_model('left_hand_rf.pkl')
        np_data=np_left
    elif(np_two_hands!=[]):
        clf=load_model('two_hands_lr.pkl')
        np_data=np_two_hands
    prediction=clf.predict(np_data).tolist()
    return jsonify(results = prediction)

@app.errorhandler(403)
def handle_no_hands_exception():
    return jsonify(error=403, message=str("There are no hands in the image"))

@app.errorhandler(404)
def handle_not_image_exception():
    return jsonify(error=403, message=str("The file is not an image"))

@app.errorhandler(404)
def handle_not_video_exception():
    return jsonify(error=403, message=str("The file is not a video"))



def is_file_type(file,type):
    mimestart=type_of_file(file.filename)
    if(mimestart==type):
        return True
    delete_file(file)
    return False
    
def type_of_file(filename):
    mimetypes.init()
    result = mimetypes.guess_type(filename)[0]
    if result != None:
        result = result.split('/')[0]
    return result

def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z
def remove_consecutive_duplicates(L):
    result=[]
    for i in range(len(L)-1):
        if(L[i]!=L[i+1]):
            result.append(L[i])
    return result
        
def return_Predictions_array(right_pred,left_pred,two_hands_pred,indexes):
    right_dict=dict(zip(indexes[0],right_pred))
    left_dict=dict(zip(indexes[1],left_pred))
    two_hands_dict=dict(zip(indexes[2],two_hands_pred))
    d=merge_two_dicts(right_dict,left_dict)
    index_pred_dict=merge_two_dicts(d,two_hands_dict)
    sorted_dict=dict(sorted(index_pred_dict.items()))
    predictions_list=[*sorted_dict.values()]
    return remove_consecutive_duplicates(predictions_list)


def load_model(filename):
    with open(filename, 'rb') as file:  
        result = pickle.load(file)
    return result

def load_models():
    right_clf = load_model('right_hand_lr.pkl')
    left_clf = load_model('left_hand_rf.pkl')
    two_hands_clf = load_model('two_hands_lr.pkl')
    return  (right_clf,left_clf,two_hands_clf)

def save_file(file):
    file.save(file.filename)

def delete_file(file):
    os.remove(file.filename)

def return_feature_map(np_images):
    result=[]
    for np_image in np_images:
        preprocessed_img=preprocess_image(np_image)
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

def preprocess_image(np_image):
    img = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
    return img_to_vector(img)


def transform_data(vect_img):#split data to right, left and both hands
    df=turn_vect_to_dataframe(vect_img)
    right_df = df.loc[df['x55'].isna()]
    right_df=right_df.loc[:,:'v54']
    left_df = df.loc[df['x34'].isna()]
    left_df=left_df[left_df.columns[np.concatenate([range(0,132),range(216,300)])]]
    two_hands_df = df.loc[~df[['x34','x55']].isna().any(axis=1)]
    indexes=[]
    indexes.append([index for index in right_df.index])
    indexes.append([index for index in left_df.index])
    indexes.append([index for index in two_hands_df.index])
    return (right_df.to_numpy(),left_df.to_numpy(),two_hands_df.to_numpy(),indexes)


def turn_vect_to_dataframe(vect_img):
    landmarks = []
    for val in range(1, 75+1):
        landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]
    df = pd.DataFrame(data=vect_img,columns=landmarks)
    return df


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
