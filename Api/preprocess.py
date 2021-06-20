import tempfile
import cv2
import numpy as np
from pathlib import Path
import mediapipe as mp
import pandas as pd

def return_feature_map(np_images):
    result=[]
    for np_image in np_images:
        preprocessed_img=preprocess_image(np_image)
        if preprocessed_img!=None:
            result.append(preprocessed_img)
    return result

def split_video_to_np_images(file,frame_rate=30):
    result=[]
    with tempfile.TemporaryDirectory() as td:
        temp_filename = Path(td) / 'file'
        file.save(temp_filename)
        vidcap = cv2.VideoCapture(str(temp_filename))
        success,image = vidcap.read()
        count = 0
        while success:
            if(count % frame_rate==0):
                rgb_image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                result.append(rgb_image)
            success,image = vidcap.read()
            count += 1
        vidcap.release()
    return result

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

def preprocess_image(np_image):
    return img_to_vector(np_image)


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
