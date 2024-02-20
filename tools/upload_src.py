import json
import os
import shutil
from pathlib import Path
from typing import Any

import click
from kaggle.api.kaggle_api_extended import KaggleApi


@click.command()
@click.option("--title", "-t", default="hms-code")
@click.option("--user_name", "-u", default="shunhiramatsu")
@click.option("--new", "-n", is_flag=True)
def main(
    title: str,
    user_name: str = "shunhiramatsu",
    new: bool = False,
):
    """extentionを指定して、dir以下のファイルをzipに圧縮し、kaggleにアップロードする。

    Args:
        title (str): kaggleにアップロードするときのタイトル
        dir (Path): アップロードするファイルがあるディレクトリ
        extentions (list[str], optional): アップロードするファイルの拡張子.
        user_name (str, optional): kaggleのユーザー名.
        new (bool, optional): 新規データセットとしてアップロードするかどうか.
    """
    # 必要に応じてコピー先ディレクトリを作成
    code_dir = "hms_code"
    #Path(code_dir).mkdir(parents=True, exist_ok=True)
    Path(code_dir + "/src").mkdir(parents=True, exist_ok=True)
    Path(code_dir + "/run").mkdir(parents=True, exist_ok=True)

    # ファイルをコピー
    src_dir = "src"
    run_dir = "run"
    shutil.copytree(src_dir, code_dir + "/src", dirs_exist_ok=True)
    shutil.copytree(run_dir, code_dir + "/run", dirs_exist_ok=True)
    #shutil.rmtree(code_dir + "/run/conf")
    
    # dataset-metadata.jsonを作成
    dataset_metadata: dict[str, Any] = {}
    dataset_metadata["id"] = f"{user_name}/{title}"
    dataset_metadata["licenses"] = [{"name": "CC0-1.0"}]
    dataset_metadata["title"] = title
    with open(code_dir + "/dataset-metadata.json", "w") as f:
        json.dump(dataset_metadata, f, indent=4)

    # api認証
    api = KaggleApi()
    api.authenticate()

    if new:
        api.dataset_create_new(
            folder=code_dir,
            dir_mode="zip",
            convert_to_csv=False,
            public=False,
        )
        print("Dataset has been created.")
    else:
        api.dataset_create_version(
            folder=code_dir,
            version_notes="",
            dir_mode="zip",
            convert_to_csv=False,
        )
        print("Dataset has been updated.")

    # delete tmp dir
    shutil.rmtree(code_dir)


if __name__ == "__main__":
    main()
