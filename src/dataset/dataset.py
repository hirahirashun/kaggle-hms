import typing as tp
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from torchvision.transforms import RandomHorizontalFlip

FilePath = tp.Union[str, Path]
Label = tp.Union[int, float, np.ndarray]

class HMSHBACDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        spec_paths: tp.Sequence[FilePath],
        eeg_spec_paths: tp.Sequence[FilePath],
        raw_eeg_paths: tp.Sequence[FilePath],
        labels: tp.Sequence[Label],
        level_labels: tp.Sequence[Label],
        spec_transform: A.Compose,
        raw_eeg_transform: tp.Any,
        use_eeg_spec: bool = True,
        use_raw_eeg: bool = False,
        is_train: bool = True,
        do_horizontal_flip: bool = False,
        do_label_smoothing: bool = False,
        num_samples: int = 10000,
        data_process_ver: int = 1,
    ):
        self.spec_paths = spec_paths
        self.eeg_spec_paths = eeg_spec_paths
        self.raw_eeg_paths = raw_eeg_paths
        self.labels = labels
        self.level_labels = level_labels

        self.spec_transform = spec_transform
        self.raw_eeg_transform = raw_eeg_transform

        self.use_eeg_spec = use_eeg_spec
        self.use_raw_eeg = use_raw_eeg

        self.is_train = is_train

        self.do_horizontal_flip = do_horizontal_flip
        self.do_label_smoothing = do_label_smoothing

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
        level_label = self.level_labels[index]

        if self.is_train and self.do_label_smoothing:
            label = self.__apply_label_smoothing(label)
            level_label = self.__apply_label_smoothing(level_label)

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
                eeg_img = [eeg_img[:, :, i:i+1] for i in range(4)]
                eeg_img = np.concatenate(eeg_img, 0)
                eeg_img = self._apply_spec_transform(eeg_img)  

                img = torch.cat([img, eeg_img], dim=0)


        if self.data_process_ver == 2:
            img = torch.cat([img, img, img], dim=0)



        horizontal_flag = False
        if self.is_train and self.do_horizontal_flip and (np.random.random() > 0.5):
            #img = RandomHorizontalFlip(p=0.5)(img)
            img = img.numpy()
            img = np.copy(img[:, :, ::-1])
            img = torch.from_numpy(img)
            horizontal_flag = True

        data_dict = {"spec_img": img, "target": label, "level_target": level_label}

        if self.use_raw_eeg:
            raw_eeg = np.load(raw_eeg_path)
            raw_eeg = self._apply_raw_eeg_transform(raw_eeg=raw_eeg)
            
            if horizontal_flag:
                raw_eeg = raw_eeg.numpy()
                raw_eeg = np.copy(raw_eeg[:, ::-1])
                raw_eeg = torch.from_numpy(raw_eeg)

            data_dict["raw_eeg"] = raw_eeg

        return data_dict

    def _apply_spec_transform(self, img: np.ndarray):
        """apply transform to image and mask"""
        transformed = self.spec_transform(image=img)
        img = transformed["image"]
        return img
    
    def _apply_raw_eeg_transform(self, raw_eeg: np.ndarray):
        raw_eeg = self.raw_eeg_transform(raw_eeg)
        samples = torch.from_numpy(raw_eeg).float()
        samples = samples.permute(1,0)

        return samples
    
    def __apply_label_smoothing(self, labels, factor=0.05):
        """
        Apply label smoothing.
        :param labels: Original labels.
        :param factor: Smoothing factor, a value between 0 and 1.
        :return: Smoothed labels.
        """
        num_classes = labels.shape[-1]
        labels = labels * (1 - factor)
        labels = labels + (factor / num_classes)
    
        return labels
    

