import logging
import os
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import yaml
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint, RichModelSummary,
                                         RichProgressBar)
from pytorch_lightning.loggers import WandbLogger

import wandb
from src.conf import TrainConfig
from src.datamodule import HMSDataModule
from src.modelmodule import HMSModel

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s:%(name)s - %(message)s"
)
LOGGER = logging.getLogger(Path(__file__).name)

@hydra.main(config_path="conf", config_name="train", version_base="1.2")
def main(cfg: TrainConfig):
    seed_everything(cfg.seed)

    is_completed_exp = False
    save_dir = cfg.dir.save_dir
    if os.path.exists(save_dir + '/best_scores.csv'):
        print(f"{cfg.exp_name} is a completed experiment.")
        best_scores_prev = pd.read_csv(save_dir + '/best_scores.csv')
        is_completed_exp = True

    os.makedirs(save_dir, exist_ok=True)

    best_scores = []
    folds = []

    for fold_id in range(cfg.n_folds):
        fold_id += cfg.n_folds_start
        os.makedirs(save_dir + f"/fold_{fold_id}", exist_ok=True)

        #if os.path.isfile(save_dir + f"/fold_{fold_id}/best_model.pth"):
        #    print("This training is already done.")
        #    continue

        # init lightning model
        datamodule = HMSDataModule(cfg, fold_id)
        LOGGER.info("Set Up DataModule")
        model = HMSModel(
            cfg, datamodule.val_df, fold_id
        )

        if is_completed_exp and (f'fold_{fold_id}' in best_scores_prev['fold_id'].values):
            model.best_score = best_scores_prev[best_scores_prev['fold_id'] == f'fold_{fold_id}']['scores'].values[0]
            print(f'Prev best score is {model.best_score}')

        # set callbacks
        checkpoint_cb = ModelCheckpoint(
            dirpath=save_dir+f"/fold_{fold_id}",
            verbose=True,
            monitor=cfg.trainer.monitor,
            mode=cfg.trainer.monitor_mode,
            save_top_k=1,
            save_last=False,
        )
        lr_monitor = LearningRateMonitor("epoch")
        progress_bar = RichProgressBar()
        early_stopping = EarlyStopping(monitor='valid_score', patience=cfg.early_stopping_rounds, )
        model_summary = RichModelSummary(max_depth=2)

        # init experiment logger
        pl_logger = WandbLogger(
            name=cfg.exp_name+f"_fold_{fold_id}",
            project="hms-harmful-brain-activity-classification",
        )
        pl_logger.log_hyperparams(cfg)

        trainer = Trainer( 
            # env
            default_root_dir=Path.cwd(),
            # num_nodes=cfg.training.num_gpus,
            accelerator=cfg.trainer.accelerator,
            devices=cfg.trainer.device,
            precision=16 if cfg.trainer.use_amp else 32,
            # training
            fast_dev_run=cfg.trainer.debug,  # run only 1 train batch and 1 val batch
            max_epochs=cfg.trainer.epochs,
            max_steps=cfg.trainer.epochs * len(datamodule.train_dataloader()),
            gradient_clip_val=cfg.trainer.gradient_clip_val,
            accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
            callbacks=[lr_monitor, progress_bar, model_summary, early_stopping],
            logger=pl_logger,
            # resume_from_checkpoint=resume_from,
            num_sanity_val_steps=0,
            log_every_n_steps=int(len(datamodule.train_dataloader()) * 0.1),
            sync_batchnorm=True,
            # check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
            check_val_every_n_epoch=None,
            val_check_interval=cfg.trainer.val_check_interval,
        )

        trainer.fit(model, datamodule=datamodule)

        best_scores.append(model.best_score)
        folds.append(f"fold_{fold_id}")
        print(f'fold_{fold_id}: best_score is {model.best_score}')

        wandb.finish()

    best_scores.append(np.mean(best_scores))
    folds.append(f"mean")

    print(f'CV score is {np.mean(best_scores)}.')
    
    """if is_completed_exp:
        best_score_df = best_scores_prev
        for i in range(len(folds)):
            best_score_df.loc[i, 'fold_id'] = np.minimum(best_scores[i], best_score_df.loc[i, 'scores'])
    else:"""
    best_score_df = pd.DataFrame(data = {"fold_id": folds, "scores": best_scores})

    best_score_df.to_csv(save_dir + '/best_scores.csv', index=False)

    return


if __name__ == "__main__":
    main()