import os
from typing import Optional

import albumentations as A
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from pytorch_lightning import LightningModule
from scipy.special import softmax
from transformers import get_cosine_schedule_with_warmup

from src.conf import TrainConfig
from src.models.common import get_model
from src.utils.augmentation import cutmix_data, mixup_data
from src.utils.kaggle_kl_div import score
from src.utils.loss_functions import get_loss_fn


class HMSModel(LightningModule):
    def __init__(
        self,
        cfg: TrainConfig,
        val_df: pd.DataFrame,
        fold_id: int
    ):
        super().__init__()
        self.cfg = cfg
        self.model = get_model(cfg)

        if self.cfg.pretrained:
            pretrained_path = f"{self.cfg.dir.output_dir}/{self.cfg.pretrained_exp}/fold_{fold_id}/best_model.pth"
            self.model.load_state_dict(torch.load(pretrained_path))

        self.loss_func = get_loss_fn(model_name=cfg.model.name, pred_confidence=cfg.pred_confidence)

        self.validation_step_outputs: list = []
        self.best_score = np.inf

        self.val_df = val_df
        self.fold_id = fold_id            


    def forward(
        self,
        x: torch.Tensor
        ):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        img, t = batch["spec_img"], batch['target']

        if self.cfg.aug.do_mixup and (np.random.random() > 0.5):
            img_original, t_original = img[:img.shape[0]//2], t[:t.shape[0]//2]
            img_mixup, t_mixup, index, lam = mixup_data(img[img.shape[0]//2:], t[t.shape[0]//2:])
            img = torch.cat([img_original, img_mixup], 0)
            t = torch.cat([t_original, t_mixup], 0)

            batch["spec_img"] = img
            batch["target"] = t

            if self.cfg.use_raw_eeg:
                eeg = batch["raw_eeg"]
                eeg_original = eeg[:eeg.shape[0]//2]
                eeg_mixup = eeg[eeg.shape[0]//2:]
                eeg_mixup = lam * eeg_mixup + (1 - lam) * eeg_mixup[index]
                eeg = torch.cat([eeg_original, eeg_mixup], 0)
                batch["raw_eeg"] = eeg

        elif self.cfg.aug.do_cutmix and (np.random.random() > 0.5):
            img_original, t_original = img[:img.shape[0]//2], t[:t.shape[0]//2]
            img_cutmix, t_cutmix, index, lam = cutmix_data(img[img.shape[0]//2:], t[t.shape[0]//2:])
            img = torch.cat([img_original, img_cutmix], 0)
            t = torch.cat([t_original, t_cutmix], 0)

            batch["spec_img"] = img
            batch["target"] = t

            if self.cfg.use_raw_eeg:
                eeg = batch["raw_eeg"]
                eeg_original = eeg[:eeg.shape[0]//2]
                eeg_mixup = eeg[eeg.shape[0]//2:]
                eeg_mixup = lam * eeg_mixup + (1 - lam) * eeg_mixup[index]
                eeg = torch.cat([eeg_original, eeg_mixup], 0)
                batch["raw_eeg"] = eeg

        output = self.model(batch)
        loss = self.loss_func(output, t)

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
                on_step=True,
                on_epoch=False,
                logger=True,
                prog_bar=True,
            )

            return loss

    def validation_step(self, batch, batch_idx):
        t = batch["target"]
        self.model.training = False
        output = self.model(batch)
        loss = self.loss_func(output, t)

        if isinstance(output, dict):
            output = output['output']


        if isinstance(loss, dict):
            for key in loss.keys():
                self.log(
                f"valid_{key}",
                loss[key],
                on_step=True,
                on_epoch=False,
                logger=True,
                prog_bar=True,
                )
            
            loss = loss['loss']

        else:
            self.log(
                "valid_loss",
                loss,
                on_step=True,
                on_epoch=False,
                logger=True,
                prog_bar=True,
            )

        self.validation_step_outputs.append(
            (   
                t.detach().cpu().numpy(),
                output.detach().cpu(),
                loss.detach().cpu().numpy(),
            )
        )

        return loss

    def on_validation_epoch_end(self):
        labels = np.concatenate([x[0] for x in self.validation_step_outputs])
        preds = np.concatenate([x[1] for x in self.validation_step_outputs])
        losses = np.array([x[2] for x in self.validation_step_outputs])
        loss = losses.mean()

        val_pred_df = pd.DataFrame(softmax(preds, axis=1), columns=self.cfg.labels)

        val_pred_df.insert(0, "label_id", self.val_df["label_id"].values)

        val_score = score(solution=self.val_df[["label_id"] + self.cfg.labels].copy().reset_index(drop=True), 
                          submission=val_pred_df, 
                          row_id_column_name='label_id')

        self.log("valid_score", val_score, on_step=False, on_epoch=True, logger=True, prog_bar=True)

        if val_score < self.best_score:
            np.save(self.cfg.dir.save_dir + f"/fold_{self.fold_id}/labels.npy", labels)
            np.save(self.cfg.dir.save_dir + f"/fold_{self.fold_id}/preds.npy", preds)
            val_pred_df.insert(0, "label_id", self.val_df["label_id"].values)
            val_pred_df.to_csv(self.cfg.dir.save_dir + f"/fold_{self.fold_id}/val_pred_df.csv", index=False)
            torch.save(self.model.state_dict(), self.cfg.dir.save_dir + f"/fold_{self.fold_id}/best_model.pth")
            print(f"Saved best model {self.best_score} -> {val_score}")
            self.best_score = val_score
        else: print("Val score is not top 1.")

        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.cfg.optimizer.lr, weight_decay=self.cfg.optimizer.weight_decay)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_training_steps=self.trainer.max_steps, **self.cfg.scheduler
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
    
    def predict_step(self, batch, batch_idx):
        t = batch["target"]
        self.model.training = False
        output = self.model(batch)
        loss = self.loss_func(output, t)

        if isinstance(output, dict):
            output = output['output']

        if isinstance(loss, dict):
            loss = loss['loss']

        self.validation_step_outputs.append(
            (   
                t.detach().cpu().numpy(),
                output.detach().cpu(),
                loss.detach().cpu().numpy(),
            )
        )

        return output
    
    def on_predict_end(self):
        labels = np.concatenate([x[0] for x in self.validation_step_outputs])
        preds = np.concatenate([x[1] for x in self.validation_step_outputs])
        losses = np.array([x[2] for x in self.validation_step_outputs])
    
        val_pred_df = pd.DataFrame(softmax(preds, axis=1), columns=self.cfg.labels)

        val_pred_df.insert(0, "label_id", self.val_df["label_id"].values)

        self.best_score = score(solution=self.val_df[["label_id"] + self.cfg.labels].copy().reset_index(drop=True), 
                          submission=val_pred_df, 
                          row_id_column_name='label_id')

        np.save(self.cfg.dir.save_dir + f"/fold_{self.fold_id}/labels.npy", labels)
        np.save(self.cfg.dir.save_dir + f"/fold_{self.fold_id}/preds.npy", preds)
        val_pred_df.insert(0, "label_id", self.val_df["label_id"].values)
        val_pred_df.to_csv(self.cfg.dir.save_dir + f"/fold_{self.fold_id}/val_pred_df.csv", index=False)

        self.validation_step_outputs.clear()

        return 