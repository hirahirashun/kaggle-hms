import typing as tp
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from scipy import signal
from torchvision.transforms import RandomHorizontalFlip

from src.utils.process_raw_eeg import (butter_bandpass_filter,
                                       butter_lowpass_filter)

FilePath = tp.Union[str, Path]
Label = tp.Union[int, float, np.ndarray]

class HMSHBACDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        spec_paths: tp.Sequence[FilePath],
        eeg_spec_paths: tp.Sequence[FilePath],
        raw_eeg_paths: tp.Sequence[FilePath],
        labels: tp.Sequence[Label],
        spec_transform: A.Compose,
        raw_eeg_transform: tp.Any,
        use_kaggle_spec: bool = True,
        use_eeg_spec: bool = True,
        use_raw_eeg: bool = False,
        use_stft_eeg: bool = False,
        is_train: bool = True,
        do_horizontal_flip: bool = False,
        do_label_smoothing: bool = False,
        do_xy_masking: bool = False,
        num_samples: int = 10000,
        data_process_ver: int = 1,
        cut_edge_spec: bool = False,
        cut_spec_width: int = 22
    ):
        self.spec_paths = spec_paths
        self.eeg_spec_paths = eeg_spec_paths
        self.raw_eeg_paths = raw_eeg_paths
        self.labels = labels

        self.spec_transform = spec_transform
        self.raw_eeg_transform = raw_eeg_transform

        self.use_kaggle_spec = use_kaggle_spec
        self.use_eeg_spec = use_eeg_spec
        self.use_raw_eeg = use_raw_eeg
        self.use_stft_eeg = use_stft_eeg

        self.is_train = is_train

        self.cut_edge_spec = cut_edge_spec
        self.cut_spec_width = cut_spec_width

        self.do_horizontal_flip = do_horizontal_flip
        self.do_label_smoothing = do_label_smoothing
        self.do_xy_masking = do_xy_masking
        if do_xy_masking:
            params = {    
            "num_masks_x": (1, 2, 3, 4),
            "num_masks_y": (1, 2, 3, 4),    
            "mask_y_length": (1, 10, 20, 30),
            "mask_x_length": (1, 10, 20, 30),
            "fill_value": 0.0,  
            }
            self.xy_transform = A.Compose([A.XYMasking(**params, p=0.5)])

        self.num_samples = num_samples

        self.data_process_ver = data_process_ver


    def __len__(self):
        #return self.num_samples if self.is_train else len(self.spec_paths)
        return len(self.spec_paths)

    def __getitem__(self, index: int):
        #if self.is_train:
        #    index = np.random.choice(len(self.spec_paths))
        
        spec_path = self.spec_paths[index]
        eeg_spec_path = self.eeg_spec_paths[index]
        raw_eeg_path = self.raw_eeg_paths[index]
        label = self.labels[index]

        if self.is_train and self.do_label_smoothing:
            label = self._apply_label_smoothing(label)
        elif np.sum(label) >= 1:
            label /= np.sum(label)

        if self.use_kaggle_spec:
            img = np.load(spec_path)  # shape: (Hz, Time) = (400, 300)
            
            # log transform
            img = np.clip(img,np.exp(-4), np.exp(8))
            img = np.log(img)
            
            # normalize per image
            eps = 1e-6
            img_mean = img.mean(axis=(0, 1))
            img = img - img_mean
            img_std = img.std(axis=(0, 1))
            img = img / (img_std + eps)

            img = img[..., None] # shape: (Hz, Time) -> (Hz, Time, Channel)
            img = self._apply_spec_transform(img)

        if self.use_eeg_spec:
            eeg_img = np.load(eeg_spec_path)         
            
            if self.data_process_ver == 1:
                eeg_img = self._apply_spec_transform(eeg_img)
                img = torch.cat([img, eeg_img], dim=0)
    
            elif self.data_process_ver == 2:
                eeg_img = [eeg_img[:, :, i:i+1] for i in range(4)]
                eeg_img = np.concatenate(eeg_img, 1)
                eeg_img = self._apply_spec_transform(eeg_img)  

                img = torch.cat([img, eeg_img], dim=2)

            elif self.data_process_ver == 3:

                if eeg_img.shape[1] == 1152:
                    eeg_img = [eeg_img[:, :, i:i+1] for i in range(4)]
                    eeg_img = np.concatenate(eeg_img, 0) # (128, 1152, 4) -> (512, 1152, 1)
                    eeg_img_0 = eeg_img[:, :640]
                    eeg_img_0 = self._apply_spec_transform(eeg_img_0)  
                    eeg_img_1 = eeg_img[:, 640:]
                    eeg_img_1 = self._apply_spec_transform(eeg_img_1)
                    eeg_img = torch.cat([eeg_img_0, eeg_img_1], dim=0)  
                else: 
                    eeg_img = [eeg_img[:, :, i:i+1] for i in range(4)]
                    eeg_img = np.concatenate(eeg_img, 0) # (128, 512, 4) -> (512, 512, 1)

                    eeg_img = self._apply_spec_transform(eeg_img)  
                if self.use_kaggle_spec:
                    img = torch.cat([img, eeg_img], dim=0)
                else: img = eeg_img


        if self.data_process_ver == 2:
            img = torch.cat([img, img, img], dim=0)



        horizontal_flag = False
        if self.is_train and self.do_horizontal_flip and (np.random.random() > 0.5):
            #img = RandomHorizontalFlip(p=0.5)(img)
            img = img.numpy()
            img = np.copy(img[:, :, ::-1])
            img = torch.from_numpy(img)
            horizontal_flag = True

        if self.is_train and self.do_xy_masking:
            img = self._apply_xy_masking(img)

        data_dict = {"spec_img": img, "target": label}

        if self.use_raw_eeg or self.use_stft_eeg:
            raw_eeg = np.load(raw_eeg_path)
            for i in range(raw_eeg.shape[-1]):
                diff_feat = raw_eeg[:, i]
                if self.is_train and (np.random.random() <= 0.1):
                    lowcut = np.random.randint(10, 20)
                    highcut = lowcut + 1.0
                    diff_feat = butter_bandpass_filter(
                        diff_feat,
                        lowcut,
                        highcut,
                        200,
                        order=2,
                    )
                if self.is_train and (np.random.random() <= 0.1):
                    diff_feat = np.zeros_like(diff_feat)

                raw_eeg[:, i] = diff_feat

            raw_eeg = np.clip(raw_eeg, -1024, 1024)

            raw_eeg = np.nan_to_num(raw_eeg, nan=0) / 32.0
            raw_eeg = raw_eeg[4000:6000, :]
            raw_eeg = butter_lowpass_filter(raw_eeg, order=2)  # 4
            raw_eeg = torch.from_numpy(raw_eeg).float()
            
            if horizontal_flag:
                raw_eeg = raw_eeg.numpy()
                raw_eeg = np.copy(raw_eeg[:, ::-1])
                raw_eeg = torch.from_numpy(raw_eeg)
            
                
            #if self.is_train:
            #    offset = ((10000 - 2000) * np.random.randint(0, 1000)) // 1000
            #else:
            #    offset = (10000 - 2000) // 2

            #raw_eeg = raw_eeg[offset:offset+2000]
                
            data_dict["raw_eeg"] = raw_eeg

            if self.use_stft_eeg:
                stft_eeg = []
                for i in range(raw_eeg.shape[-1]):
                    f, t, linear = signal.stft(raw_eeg[:, i], fs=200, nperseg=400, noverlap=350)
                    this_inputs = torch.FloatTensor(linear[:40]).transpose(0,1) #.to('cuda')
                    this_inputs = this_inputs[:, :, None]
                    stft_eeg.append(this_inputs)
                stft_eeg = torch.cat(stft_eeg, axis=-1)
                
                data_dict['stft_eeg'] = stft_eeg

        return data_dict

    def _apply_spec_transform(self, img: np.ndarray):
        """apply transform to image and mask"""
        transformed = self.spec_transform(image=img)
        img = transformed["image"]
        return img
    
    def _apply_raw_eeg_transform(self, raw_eeg: np.ndarray):
        # raw_eeg = self.raw_eeg_transform(raw_eeg)
        samples = torch.from_numpy(raw_eeg).float()
        samples = samples.permute(1,0)

        return samples
    
    def _apply_label_smoothing(self, labels, factor=0.05):
        """
        Apply label smoothing.
        :param labels: Original labels.
        :param factor: Smoothing factor, a value between 0 and 1.
        :return: Smoothed labels.
        """
        num_classes = labels.shape[-1]
        label_sum = np.sum(labels)
        factor =  1/(5 + label_sum)

        labels /= label_sum
        labels = labels * (1 - factor)
        labels = labels + (factor / num_classes)
    
        return labels
    
    
    def _apply_xy_masking(self, img):
        img = torch.permute(img, (1, 2, 0))
        img = img.numpy()
        img = self.xy_transform(image=img)["image"]
        img = torch.from_numpy(img)
        img = torch.permute(img, (2, 0, 1))
    
        return img

