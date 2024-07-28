#coding:utf-8
import math
import sys
import warnings
import numpy as np
from scipy.io import loadmat,savemat
from scipy.signal import butter, lfilter
warnings.filterwarnings("ignore")
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
#  假设采样频率为400hz,信号本身最大的频率为200hz，要滤除0.5hz以下，50hz以上频率成分，即截至频率为0.5hz，50hz
from matplotlib.colors import Normalize

def butter_bandpass_filter(data, lowcut, highcut, samplingRate, order=5):
	nyq = 0.5 * samplingRate
	low = lowcut / nyq
	high = highcut / nyq
	b, a = butter(order, [low, high], btype='band')
	y = lfilter(b, a, data)
	return y

def compute_DE(data):
    variance = np.var(data, ddof=1)
    return math.log(2 * math.pi * math.e * variance) / 2


def decompose(filepath):
    #
    data = loadmat(filepath)['EEG'][0][0][0]
    frequency = 200
    samples = data.shape[0]
    channels = data.shape[1]

    # 100个采样点计算一个微分熵
    num_sample = int(samples/100)

    bands = 5
    # 微分熵特征[14160, 17, 5]
    DE_Characteristics = np.empty([num_sample, channels, bands])

    temp_de = np.empty([0, num_sample])

    # 分别计算每个频道的微分熵
    for channel in range(channels):

        trail_single = data[:, channel]

        Delta = butter_bandpass_filter(trail_single, 0.5, 4, frequency, order=3)
        Theta = butter_bandpass_filter(trail_single, 4, 8, frequency, order=3)
        Alpha = butter_bandpass_filter(trail_single, 8, 12, frequency, order=3)
        Beta = butter_bandpass_filter(trail_single, 12, 30, frequency, order=3)
        Gamma = butter_bandpass_filter(trail_single, 30, 50, frequency, order=3)


        DE_Delta = np.zeros(shape=[0], dtype=float)
        DE_Theta = np.zeros(shape=[0], dtype=float)
        DE_alpha = np.zeros(shape=[0], dtype=float)
        DE_beta = np.zeros(shape=[0], dtype=float)
        DE_gamma = np.zeros(shape=[0], dtype=float)


        # 依次计算5个频带的微分熵
        for index in range(num_sample):
            DE_Delta = np.append(DE_Delta, compute_DE(Delta[index * 100: (index + 1) * 100]))
            DE_Theta = np.append(DE_Theta, compute_DE(Theta[index * 100: (index + 1) * 100]))
            DE_alpha = np.append(DE_alpha, compute_DE(Alpha[index * 100: (index + 1) * 100]))
            DE_beta = np.append(DE_beta, compute_DE(Beta[index * 100: (index + 1) * 100]))
            DE_gamma = np.append(DE_gamma, compute_DE(Gamma[index * 100: (index + 1) * 100]))


        temp_de = np.vstack([temp_de, DE_Delta])
        temp_de = np.vstack([temp_de, DE_Theta])
        temp_de = np.vstack([temp_de, DE_alpha])
        temp_de = np.vstack([temp_de, DE_beta])
        temp_de = np.vstack([temp_de, DE_gamma])
        print("_____________________")


    temp_trail_de = temp_de.reshape(-1, 5, num_sample)
    print("trail_DE shape", DE_Characteristics.shape)
    temp_trail_de = temp_trail_de.transpose([2, 0, 1])
    DE_Characteristics = np.vstack([temp_trail_de])

    """
        14160*17*6: 885*16(每个样本分为16段采样)*17*5
    """
    print("trail_DE shape", DE_Characteristics.shape)
    return DE_Characteristics





def loadEOG(EOG):
    ica=EOG['features_table_ica']
    minus=EOG['features_table_minus']
    minh=EOG['features_table_icav_minh']
    out=np.stack((ica, minus, minh))
    return out
def function_test():
    eegFile = "data/save/EEG/"
    eogFile="data/SEED-VIG/EOG_Feature/"
    datafiles=os.listdir(eegFile)
    EOG_List=[]
    EEG_List=[]
    for file in datafiles:
        eeg_path=eegFile+file
        eog_path=eogFile+file
        try:
            eeg = loadmat(eeg_path)["EEG"]
            eog = loadmat(eog_path)
            eog = loadEOG(eog)
            eog =eog.transpose(1,0,2)

            EEG_List.append(eeg)
            EOG_List.append(eog)

        except Exception:
            print("错误:"+file)
            print(Exception)

    EEGDATA=np.concatenate(EEG_List)
    EOGDATA=np.concatenate(EOG_List)
    path='data/save/data/'
    np.save(path+'EEG_DATA.npy',EEGDATA)
    np.save(path+'EOG_DATA.npy',EOGDATA)



def Normal_mat(data):
    min_val = np.min(data)
    max_val = np.max(data)

    # 归一化数组
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data
def cutSegments(raw,duration,rate,channel_labels):
    length=raw.shape[1]
    channels=raw.shape[0]
    n_samples = int(duration *rate)
    start_indices = np.arange(0, length - n_samples + 1, n_samples)
    segments = []
    frequency = 200
    images=[]
    for start in tqdm(start_indices):
        segment = raw[:,start:start + n_samples]


        channel_data=[]
        for i in range(channels):
            signal_data=segment[i]
            Delta = butter_bandpass_filter(signal_data, 0.5, 4, frequency, order=3)
            Theta = butter_bandpass_filter(signal_data, 4, 8, frequency, order=3)
            Alpha = butter_bandpass_filter(signal_data, 8, 12, frequency, order=3)
            Beta = butter_bandpass_filter(signal_data, 12, 30, frequency, order=3)
            Gamma = butter_bandpass_filter(signal_data, 30, 50, frequency, order=3)

            DE_Delta = np.zeros(shape=[0], dtype=float)
            DE_Theta = np.zeros(shape=[0], dtype=float)
            DE_alpha = np.zeros(shape=[0], dtype=float)
            DE_beta = np.zeros(shape=[0], dtype=float)
            DE_gamma = np.zeros(shape=[0], dtype=float)

            num_rate = 100
            num_sample = int(n_samples / num_rate)
            # 依次计算5个频带的微分熵
            for index in range(num_sample):
                DE_Delta = np.append(DE_Delta, compute_DE(Delta[index * num_rate: (index + 1) * num_rate]))
                DE_Theta = np.append(DE_Theta, compute_DE(Theta[index * num_rate: (index + 1) * num_rate]))
                DE_alpha = np.append(DE_alpha, compute_DE(Alpha[index * num_rate: (index + 1) * num_rate]))

            DE_Delta = np.pad(DE_Delta, (0, 16), mode='constant')
            DE_Theta = np.pad(DE_Theta, (0, 16), mode='constant')
            DE_alpha = np.pad(DE_alpha, (0, 16), mode='constant')

            num_rate = 50
            num_sample = int(n_samples / num_rate)
            for index in range(num_sample):
                DE_alpha = np.append(DE_alpha, compute_DE(Alpha[index * num_rate: (index + 1) * num_rate]))
                DE_beta = np.append(DE_beta, compute_DE(Beta[index * num_rate: (index + 1) * num_rate]))
                DE_gamma = np.append(DE_gamma, compute_DE(Gamma[index * num_rate: (index + 1) * num_rate]))
            cha = np.stack([DE_alpha, DE_beta, DE_gamma])
            channel_data.append(cha)
        seg=np.stack(channel_data)
        images.append(seg)

    return np.stack(images)
if __name__=='__main__':
    # function_one()
    # decompose("data/10_20151125_noon.mat")
    # function_test()
    # eegFile = "data/SEED-VIG/Raw_Data/"
    # saveFile="data/save/EEG/"
    # datafiles = os.listdir(eegFile)
    # for file in tqdm(datafiles):
    #     try:
    #         de=decompose(eegFile+file)
    #         de=np.reshape(de, (885, 16, 17, 5))
    #         de = de.transpose(0, 2, 1, 3)
    #         de = np.reshape(de, (885, 17, -1))
    #         savemat(saveFile+file,{"EEG":de})
    #     except Exception:
    #         print("错误:"+Exception)




    data_path="data/Raw_Data/"
    datafiles = os.listdir(data_path)

    Label_list=[]
    segments=[]
    for i,file in enumerate(datafiles):
        raw=loadmat(data_path+file)
        EEG_data=raw['EEG'][0][0][0]
        channel_labels=raw['EEG'][0][0][1]
        EEG_data=EEG_data.transpose()
        segment=cutSegments(EEG_data,8,200,channel_labels)
        segments.append(segment)

    EEG_DATA=np.concatenate(segments,axis=0)
    path = 'data/origin/'
    np.save(path + 'EEG_DATA.npy', EEG_DATA)
    # print(EEG_DATA.shape)

    # data_path="Raw_Data/"
    #
    # label_path = "EOG_Feature/"
    # datafiles = os.listdir(data_path)
    # eog_list=[]
    # for i,file in enumerate(datafiles):
    #     raw=loadmat(label_path+file)
    #     EOG_ica=raw["features_table_ica"]
    #     EOG_minus=raw["features_table_minus"]
    #     EOG_minh=raw["features_table_icav_minh"]
    #     EOG_data=np.stack([EOG_ica,EOG_minus,EOG_minh],axis=1)
    #     eog_list.append(EOG_data)
    #
    # EEG_DATA=np.concatenate(eog_list,axis=0)
    # path = 'data/origin/'
    # np.save(path + 'EOG_DATA.npy', EEG_DATA)
    # print(EEG_DATA.shape)


