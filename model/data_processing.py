from torch.utils.data import Dataset
import torch
import numpy as np
class CurrentDataSet(Dataset):
    def __init__(self,data_path):
        self.EEG_DATA=np.load(data_path+'EEG_DATA.npy')
        self.EOG_DATA = np.load(data_path + 'EOG_DATA.npy')
        self.LABEL_DATA=np.load(data_path+'LABEL_DATA.npy')
    def __len__(self):
        return len(self.LABEL_DATA)
    def getClassWeight(self):
        pass
    def __getitem__(self, index):
        EEG=self.EEG_DATA[index]
        EOG=self.EOG_DATA[index]
        LABEL=self.LABEL_DATA[index]
        EEG=torch.tensor(EEG, dtype=torch.float32)
        EOG=torch.tensor(EOG, dtype=torch.float32)
        LABEL=torch.tensor(LABEL, dtype=torch.float32)
        return EEG,EOG,LABEL
