import glob
import os
import pickle
import re
from river import metrics

def load_model(path, model_timestamp):
    if model_timestamp is None:
        model_path = max(glob.iglob(f'{path}/*.pickle'), key=os.path.getctime)
        model_timestamp = re.search(r"model_([0-9]+)\.pickle", model_path).group(1)
    else:
        model_path = f"{path}/model_{model_timestamp}.pickle"
    with open(model_path, 'rb') as f:
        model=pickle.load(f)
    return (model, model_timestamp)




def test(dataset, N:int, subpath:str='default', model=None, timestamp=None):    
    if model is None:
        model_path=f'models/{subpath}'
        (model, timestamp) = load_model(model_path, timestamp)
   
    acc= metrics.Accuracy()
    prec_attack= metrics.Precision()
    recl_attack= metrics.Recall()
    prec_normal= metrics.Precision()
    recl_normal= metrics.Recall()
    f1= metrics.F1()
    i = 0   
    done = False
    ind = dataset.ind
    while not done:
        for (X, Y) in dataset:
            i += 1
            if N > 0 and i >= N:
                done = True
                break;

            Yp = model.predict_one(X)
            
            acc.update(Y, Yp)
            prec_attack.update(Y, Yp)
            recl_attack.update(Y, Yp)
            prec_normal.update(Y + 1 % 2, Yp + 1 % 2)
            recl_normal.update(Y + 1 % 2, Yp + 1 % 2)
            f1.update(Y, Yp)
      
        
        if not done and N > 0:
            print("Test dataset reached end. Resetting.")
            dataset.reset(ind)
        else:
            done = True
    
    return (acc, prec_attack, recl_attack, prec_normal, recl_normal, f1)
