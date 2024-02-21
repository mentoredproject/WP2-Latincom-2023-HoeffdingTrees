from .train import train, load_stats, plot_stats
from .test import test
from .plot import plot_stats, regenerate_plots
import os
import pandas as pd

class Dataset():

    #Initializes your dataset
    def __init__(self):    
        pass;       

    def __iter__(self):
        pass;

    def reset(self):
        pass;

class DummyPredictor:
    def __init__(self, label=1):
        self.label=label
    
    def learn_one(self,  x, y):
        pass

    def predict_one(self, x):
        return self.label