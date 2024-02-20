import os

import click
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold
from sklearn.utils import shuffle


@click.command()
@click.option("--random_state", "-r", default=0)
@click.option("--method", "-m", default="gkf")
@click.option("--version", "-v", default=6)
@click.option("--n_folds", "-f", default=5)
@click.option("--split_by", "-s", default="eeg_id")
def main(random_state, method, version, n_folds, split_by):
    train = pd.read_csv("/home/hiramatsu/kaggle/hms-harmful-brain-activity-classification/data/train.csv")
    train = train.groupby(split_by).head(1).reset_index(drop=True)

    train = shuffle(train, random_state=random_state)

    if method == "gkf":
        sgkf = GroupKFold(n_splits=n_folds)
    elif method == "sgkf":
        sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=0)

    #kf = KFold(n_splits=n_folds, shuffle=True)

    fold_path = f'/home/hiramatsu/kaggle/hms-harmful-brain-activity-classification/folds/{split_by}_fold{n_folds}_v{version}'
    os.makedirs(fold_path, exist_ok=True)

    config = {"random_state": random_state, "method": method}
    with open(fold_path + '/config.yaml', 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    for fold_id, (train_idx, val_idx) in enumerate(sgkf.split(X=train, y=train['expert_consensus'], groups=train['patient_id'])):
        train_ids = train[split_by].values[train_idx]
        val_ids = train[split_by].values[val_idx]  
        folds_file = {}
        folds_file['train_id'] = train_ids.tolist()
        folds_file['val_id'] = val_ids.tolist()

        with open(fold_path + f'/fold_{fold_id}.yaml', 'w') as outfile:
            yaml.dump(folds_file, outfile, default_flow_style=False)
        print(fold_id, len(train[train[split_by].isin(train_ids)]), len(train[train[split_by].isin(val_ids)]))


if __name__ == "__main__":
    main()