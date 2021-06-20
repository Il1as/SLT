import pickle

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
    right_clf = load_model('right_hand_logReg.pkl')
    left_clf = load_model('left_hand_randForest.pkl')
    two_hands_clf = load_model('two_hands_logReg.pkl')
    return  (right_clf,left_clf,two_hands_clf)