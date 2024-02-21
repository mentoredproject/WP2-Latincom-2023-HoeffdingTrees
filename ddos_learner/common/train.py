import numpy as np
from river import utils, metrics
from .test import test
from .plot import plot_stats
import matplotlib.pyplot as plt
from time import time
import pickle
import os
import glob
import re
from datetime import datetime
import pandas as pd

def save_model(model, path, timestamp):
    

    if not os.path.exists(path):
        print(path)
        os.makedirs(path)

    with open(f'{path}/model_{timestamp}.pickle', 'wb') as f:
        pickle.dump(model, f)

def save_stats(path, idx, timestamp, acc, prec_attack, recl_attack, prec_normal, recl_normal, f1, lb, timest, qnt_hist):

    if not os.path.exists(path):        
        os.makedirs(path)

    df = pd.DataFrame.from_dict({'idx':idx, 'timestamp': timestamp, 'acc':acc, 'prec_attack':prec_attack, 'recl_attack':recl_attack, 'prec_normal':prec_normal, 'recl_normal':recl_normal, 'f1':f1, 'lb':lb, 'timest':timest, 'qnt':qnt_hist})
    df.to_csv(f'{path}/{timestamp}.csv', index=False)   

def load_stats(path, timestamp=None):
    if timestamp is None:
        path = max(glob.iglob(f'{path}/*.pickle'), key=os.path.getctime)
        timestamp = re.search(r"model_([0-9]+)\.pickle", path).group(1)
    else:
        path = f"{path}/model_{timestamp}.pickle"
    with open(path, 'rb') as f:
        stats=pickle.load(f)
    return (stats, timestamp)




def train(model, dataset , N:int, step_size:int = 100, subpath:str='default' ):    
    
    print("Training initiated... ")

    it = iter(dataset)

    acc_hist = np.empty(N//step_size + 1)
    prec_attack_hist = np.empty(N//step_size + 1)
    recl_attack_hist = np.empty(N//step_size + 1)
    prec_normal_hist = np.empty(N//step_size + 1)
    recl_normal_hist = np.empty(N//step_size + 1)
    f1_hist = np.empty(N//step_size + 1)
    lb_hist = np.empty(N//step_size + 1)
    timest_hist=np.empty(N//step_size + 1)
    qnt_hist=np.empty(N//step_size + 1)    
    start_time=time()

    lb=0

    i = 0
    done = False
    while not done:
        for (X, Y) in dataset:
            lb += Y
            i += 1
            if N > 0 and i >= N:
                done = True
                break;
            Yp = model.predict_one(X)     

            model.learn_one(X, Y)

            train_start = time()
            if i%step_size == 0:
                print(f"At {i}.")
                dataset.reset(i)
                (acc, prec_attack, recl_attack, prec_normal, recl_normal, f1) = test(dataset, step_size*4, model=model)
                
                dataset.reset(i)
                acc_hist[i//step_size] = acc.get()
                prec_attack_hist[i//step_size] = prec_attack.get()
                recl_attack_hist[i//step_size] = recl_attack.get()
                prec_normal_hist[i//step_size] = prec_normal.get()
                recl_normal_hist[i//step_size] = recl_normal.get()
                f1_hist[i//step_size] = f1.get()

                
                lb_hist[i//step_size] = lb / step_size
                timest_hist[i//step_size] = time() - start_time
                qnt_hist[i//step_size] = i
                lb = 0
                print(f"Acc: {acc.get():.8f}. Prec(Attack): {prec_attack.get():.8f}.\nRecl (Attack): {recl_attack.get():.8f}. Prec(Normal): {prec_normal.get():.8f}.\nRecl (Normal): {recl_normal.get():.8f}. F1: {f1.get():.8f}. LB: {lb_hist[i//step_size]:.8f}")
            start_time += time() - train_start
        if not done and N > 0:
            print("Resetting dataset.")
            dataset.reset()
        else:
            done = True

    timestamp = int((datetime.now() - datetime(1970, 1, 1)).total_seconds())

    model_path = f'models/{subpath}'

    save_model(model, model_path, timestamp)

    stats_path=f'stats/{subpath}'

    idx=np.arange(1, N, step_size)

    save_stats(stats_path, idx, timestamp, acc_hist, prec_attack_hist, recl_attack_hist, prec_normal_hist, recl_normal_hist, f1_hist, lb_hist, timest_hist, qnt_hist)    

    plot_path=f'plots/{subpath}'

    plot_stats(plot_path, idx, timestamp, acc_hist, prec_attack_hist, recl_attack_hist, prec_normal_hist, recl_normal_hist, f1_hist, lb_hist, timest_hist, qnt_hist)



