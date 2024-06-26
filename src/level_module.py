from typing import Optional

import albumentations as A
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning import LightningModule
from sklearn.metrics import average_precision_score
from transformers import get_cosine_schedule_with_warmup

from src.conf import TrainConfig
from src.models.common import get_model
from src.utils.augmentation import cutmix_data, mixup_data
from src.utils.loss_functions import (KLDivLossWithLogits,
                                      KLDivLossWithLogitsForVal,
                                      KLDWithContrastiveLoss)


class HMSLevelModel(LightningModule):
    def __init__(
        self,
        cfg: TrainConfig,
        val_df: pd.DataFrame,
        fold_id: int
    ):
        super().__init__()
        self.cfg = cfg

        if cfg.data_process_ver == 2:
            in_channels = 3
        elif cfg.data_process_ver == 1:
            in_channels = 1
            if cfg.use_eeg_spec:
                in_channels += 4
        elif cfg.data_process_ver == 3:
            in_channels = 1
            if cfg.use_eeg_spec:
                in_channels += 1
            
        self.model = timm.create_model(model_name="efficientnet_b0", pretrained=True, in_chans=in_channels, num_classes=4, drop_rate=0.2, drop_path_rate=0.2)

        self.loss_func = nn.BCEWithLogitsLoss() 

        self.validation_step_outputs: list = []
        self.best_score = 0.0

        self.val_df = val_df
        self.fold_id = fold_id            


    def forward(
        self,
        x: torch.Tensor
        ):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        img, t, level_t = batch["spec_img"],  batch["target"], batch['level_target']

        if self.cfg.aug.do_mixup and (np.random.random() > 0.5):
            img_original, t_original, level_t_original = img[:img.shape[0]//2], t[:t.shape[0]//2], level_t[:level_t.shape[0]//2]
            img_mixup, t_mixup, level_t_mixup, index, lam = mixup_data(img[img.shape[0]//2:], t[t.shape[0]//2:], level_t[level_t.shape[0]//2:])
            img = torch.cat([img_original, img_mixup], 0)
            t = torch.cat([t_original, t_mixup], 0)
            level_t = torch.cat([level_t_original, level_t_mixup])
            batch["spec_img"] = img
            batch["target"] = t
            batch['level_target'] = level_t

            if self.cfg.use_raw_eeg:
                eeg = batch["raw_eeg"]
                eeg_original = eeg[:eeg.shape[0]//2]
                eeg_mixup = eeg[eeg.shape[0]//2:]
                eeg_mixup = lam * eeg_mixup + (1 - lam) * eeg_mixup[index]
                eeg = torch.cat([eeg_original, eeg_mixup], 0)
                batch["raw_eeg"] = eeg

        elif self.cfg.aug.do_cutmix and (np.random.random() > 0.5):
            img_original, t_original, level_t_original = img[:img.shape[0]//2], t[:t.shape[0]//2], level_t[:level_t.shape[0]//2]
            img_cutmix, t_cutmix, level_t_cutmix, index, lam = cutmix_data(img[img.shape[0]//2:], t[t.shape[0]//2:], level_t[level_t.shape[0]//2:])
            img = torch.cat([img_original, img_cutmix], 0)
            t = torch.cat([t_original, t_cutmix], 0)
            level_t = torch.cat([level_t_original, level_t_cutmix])
            batch["spec_img"] = img
            batch["target"] = t
            batch["level_target"] = level_t

            if self.cfg.use_raw_eeg:
                eeg = batch["raw_eeg"]
                eeg_original = eeg[:eeg.shape[0]//2]
                eeg_mixup = eeg[eeg.shape[0]//2:]
                eeg_mixup = lam * eeg_mixup + (1 - lam) * eeg_mixup[index]
                eeg = torch.cat([eeg_original, eeg_mixup], 0)
                batch["raw_eeg"] = eeg

        output = self.model(img)
        loss = self.loss_func(output, level_t)

        if isinstance(loss, dict):
            for key in loss.keys():
                self.log(
                f"train_{key}",
                loss[key],
                on_step=False,
                on_epoch=True,
                logger=True,
                prog_bar=True,
                )
            
            return loss['loss']

        else:
            self.log(
                "train_loss",
                loss,
                on_step=False,
                on_epoch=True,
                logger=True,
                prog_bar=True,
            )

            return loss

    def validation_step(self, batch, batch_idx):
        t = batch["target"]
        level_t = batch['level_target']
        self.model.training = False
        output = self.model(batch['spec_img'])
        loss = self.loss_func(output, level_t)

        if isinstance(output, dict):
            output = output['weighted_output']


        if isinstance(loss, dict):
            for key in loss.keys():
                self.log(
                f"valid_{key}",
                loss[key],
                on_step=False,
                on_epoch=True,
                logger=True,
                prog_bar=True,
                )
            
            loss = loss['loss']

        else:
            self.log(
                "valid_loss",
                loss,
                on_step=False,
                on_epoch=True,
                logger=True,
                prog_bar=True,
            )

        self.validation_step_outputs.append(
            (   
                level_t.detach().cpu().numpy(),
                output.softmax(dim=1).detach().cpu(),
                loss.detach().cpu().numpy(),
            )
        )

        return loss

    def on_validation_epoch_end(self):
        labels = np.concatenate([x[0] for x in self.validation_step_outputs])
        preds = np.concatenate([x[1] for x in self.validation_step_outputs])
        losses = np.array([x[2] for x in self.validation_step_outputs])
        loss = losses.mean()

        val_pred_df = pd.DataFrame(preds, columns=["pattern_edge", "pattern_idealized", "pattern_proto", "pattern_undecided"])

        val_pred_df.insert(0, "label_id", self.val_df["label_id"].values)

        score = average_precision_score(labels, preds)
        self.log("valid_score", score, on_step=False, on_epoch=True, logger=True, prog_bar=True)


        if score > self.best_score:
            np.save(self.cfg.dir.save_dir + f"/fold_{self.fold_id}/labels.npy", labels)
            np.save(self.cfg.dir.save_dir + f"/fold_{self.fold_id}/preds.npy", preds)
            val_pred_df.to_csv(self.cfg.dir.save_dir + f"/fold_{self.fold_id}/val_pred_df.csv", index=False)
            torch.save(self.model.state_dict(), self.cfg.dir.save_dir + f"/fold_{self.fold_id}/best_model.pth")
            print(f"Saved best model {self.best_score} -> {score}")
            self.best_score = score

        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.cfg.optimizer.lr)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_training_steps=self.trainer.max_steps, **self.cfg.scheduler
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]