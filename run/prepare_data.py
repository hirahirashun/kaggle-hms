import logging
import os
from pathlib import Path

import click
import librosa
import numpy as np
import pandas as pd
import pywt
from tqdm.auto import tqdm

USE_WAVELET = None

NAMES = ['LL','LP','RP','RR']

FEATS = [['Fp1','F7','T3','T5','O1'],
         ['Fp1','F3','C3','P3','O1'],
         ['Fp2','F8','T4','T6','O2'],
         ['Fp2','F4','C4','P4','O2']]

# DENOISE FUNCTION
def maddest(d, axis=None):
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)

def denoise(x, wavelet='haar', level=1):    
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1/0.6745) * maddest(coeff[-level])

    uthresh = sigma * np.sqrt(2*np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])

    ret=pywt.waverec(coeff, wavelet, mode='per')
    
    return ret

def spectrogram_from_eeg(eeg, spec_width=256, n_mels=128):    
    # VARIABLE TO HOLD SPECTROGRAM
    img = np.zeros((n_mels,spec_width,4),dtype='float32')

    signals = []
    for k in range(4):
        COLS = FEATS[k]
        
        for kk in range(4):
            # COMPUTE PAIR DIFFERENCES
            x = eeg[COLS[kk]].values - eeg[COLS[kk+1]].values

            # FILL NANS
            m = np.nanmean(x)
            if np.isnan(x).mean()<1: x = np.nan_to_num(x,nan=m)
            else: x[:] = 0

            # DENOISE
            if USE_WAVELET:
                x = denoise(x, wavelet=USE_WAVELET)
            signals.append(x)

            # RAW SPECTROGRAM
            mel_spec = librosa.feature.melspectrogram(y=x, sr=200, hop_length=len(x)//spec_width, 
                  n_fft=1024, n_mels=n_mels, fmin=0, fmax=20, win_length=n_mels)

            
            # LOG TRANSFORM
            width = (mel_spec.shape[1]//32)*32

            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)[:,:width]

            # STANDARDIZE TO -1 TO 1
            mel_spec_db = (mel_spec_db+40)/40 
            img[:,:,k] += mel_spec_db
                
        # AVERAGE THE 4 MONTAGE DIFFERENCES
        img[:,:,k] /= 4.0 # (Hz, time)
        
        
    return img

@click.command()
@click.option("--phase", "-p", default="train")
def main(phase):
    data_path = "/home/hiramatsu/kaggle/hms-harmful-brain-activity-classification/data" if phase=='train' else "/kaggle/input/hms-harmful-brain-activity-classification"
    eeg_path = data_path + f"/{phase}_eegs"
    spec_path =  data_path + f"/{phase}_spectrograms"
    save_path = "/home/hiramatsu/kaggle/hms-harmful-brain-activity-classification/outputs" if phase=='train' else "/kaggle/working"
    save_raw_eeg_path = save_path + f"/{phase}_raw_eeg_split"
    save_eeg_spec_path = save_path + f"/{phase}_eeg_spectrograms_split"
    save_large_eeg_spec_path = save_path + f"/{phase}_eeg_spectrograms_v2_split"
    save_spec_path = save_path + f"/{phase}_spectrograms_split"
    

    df = pd.read_csv(data_path + f"/{phase}.csv")

    if phase == 'test':
        df["eeg_label_offset_seconds"] = 0
        df["spectrogram_label_offset_seconds"] = 0
        df['label_id'] = df['eeg_id']

    df["eeg_label_offset_seconds"] = df["eeg_label_offset_seconds"].astype(int)
    df["spectrogram_label_offset_seconds"] = df["spectrogram_label_offset_seconds"].astype(int)

    os.makedirs(save_spec_path, exist_ok=True)
    for spec_id, this_df in tqdm(df.groupby("spectrogram_id")):
        spec = pd.read_parquet(spec_path + f"/{spec_id}.parquet")
        
        spec_arr = spec.fillna(0).values[:, 1:].T.astype("float32")  # (Hz, Time) = (400, 300)
        
        # label_idに対応するスペクトログラムを抽出して保存している
        for spec_offset, label_id in this_df[
            ["spectrogram_label_offset_seconds", "label_id"]
        ].astype(int).values:
            spec_offset = spec_offset // 2 
            split_spec_arr = spec_arr[:, spec_offset: spec_offset + 300]
            np.save(save_spec_path + f"/{int(label_id)}.npy" , split_spec_arr)

    os.makedirs(save_raw_eeg_path, exist_ok=True)
    os.makedirs(save_eeg_spec_path, exist_ok=True)
    os.makedirs(save_large_eeg_spec_path, exist_ok=True)

    FEATS = ['Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'Fz', 'Cz', 'Pz',
       'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2', 'EKG']

    for eeg_id, this_df in tqdm(df.groupby("eeg_id")):
        eeg = pd.read_parquet(f'{eeg_path}/{eeg_id}.parquet')
        
        #this_df = this_df[this_df['eeg_id'] == eeg_id]

        for eeg_offset, label_id in this_df[
            ["eeg_label_offset_seconds", "label_id"]
        ].values:
            eeg_offset *= 200
            eeg_offset = int(eeg_offset)

            this_eeg = eeg[eeg_offset:eeg_offset+10_000]

            data = np.zeros((10_000,len(FEATS)))
            for j,col in enumerate(FEATS):
                
                # FILL NAN
                x = this_eeg.loc[:, col].values.astype("float32")
                m = np.nanmean(x)
                if np.isnan(x).mean()<1: x = np.nan_to_num(x,nan=m)
                else: x[:] = 0
                    
                data[:,j] = x

            np.save(save_raw_eeg_path + f"/{int(label_id)}.npy" , data)
            img = spectrogram_from_eeg(this_eeg)
            np.save(save_eeg_spec_path + f"/{int(label_id)}.npy" , img)
            large_img = spectrogram_from_eeg(this_eeg, spec_width=512, n_mels=128) #(128, 512, 4)
            np.save(save_large_eeg_spec_path + f"/{int(label_id)}.npy" , large_img)


if __name__ == "__main__":
    main()