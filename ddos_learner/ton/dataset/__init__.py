from ...common import Dataset
from river import stream

class TONIoTDataset(Dataset):
    
    def __init__(self, train=True):        
        self.path=f"ddos_learner/ton/dataset/processed/train_data.csv"
        self.iter=stream.iter_csv(self.path, converters={'duration': float,'src_bytes': float,'dst_bytes':float,'missed_bytes':float,'src_pkts':float,'dst_pkts':float,'label':float}, target='label')
        self.data=[d for d in self.iter]
        self.ind=0
        self.reset()

    def __iter__(self):
        while self.ind < len(self.data):
            yield self.data[self.ind]
            self.ind += 1

    
    def reset(self, ind=0):
        self.ind=ind
