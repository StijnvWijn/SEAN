"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
from collections import OrderedDict

import data
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from util.visualizer import Visualizer
from util import html
import yaml
from pathlib import Path
from qcardia_data import DataModule
import wandb
from copy import deepcopy
import nibabel as nib
import numpy as np
import pandas as pd
import shutil
from qcardia_data.utils import sample_from_csv_by_group, data_to_file


def main():
    opt = TestOptions().parse()
    opt.status = 'test'
    # load the dataset
    config_path = Path("/home/bme001/20183502/code/msc-stijn/resources/example-config_original.yaml")

    # The config contains all the model hyperparameters and training settings for the
    # experiment. Additionally, it contains data preprocessing and augmentation
    # settings, paths to data and results, and wandb experiment parameters.
    config = yaml.load(open(config_path), Loader=yaml.FullLoader)

    for key in config['data']['augmentation']:
        if 'prob' in config['data']['augmentation'][key]:
            if config['data']['augmentation'][key]['prob'] != 0.0:
                print(f"WARNING: Setting augmentation {key} prob to 0.0")
            config['data']['augmentation'][key]['prob'] = 0.0
        if key == 'flip_prob':
            print(f"WARNING: Setting augmentation {key} to [0.0,0.0]")
            config['data']['augmentation'][key] = [0.0, 0.0]

    run = wandb.init(
            project=config["experiment"]["project"],
            name=config["experiment"]["name"],
            config=config,
            save_code=True,
            mode="disabled",
        )

    # Get the path to the directory where the Weights & Biases run files are stored.
    online_files_path = Path(run.dir)
    print(f"online_files_path: {online_files_path}")

    datasets = []
    #Split datasets if multiple different values for key_pairs are given
    if type(wandb.config["dataset"]['subsets']) == dict:
        unique_keys = []
        unique_datasets = []
        for key, value in wandb.config["dataset"]['subsets'].items():
            if value[0] not in unique_keys:
                unique_keys.extend(value)
                unique_datasets.append([key])
            else:
                unique_datasets[unique_keys.index(value[0])].append(key)
        data_config = deepcopy(wandb.config.as_dict())
        for i in range(len(unique_datasets)):
            if '=meta' in unique_keys[i][1]:
                unique_keys[i][1] = unique_keys[i][1].split('=')[0]
                data_config['dataset']['meta_only_labels'] = True
                print(f"Meta only labels datasets {unique_datasets[i]} with keys {unique_keys[i]}")
            elif str(unique_keys[i][1]).lower() in ['none', 'null', '']:
                data_config['dataset']['meta_only_labels'] = False
                unique_keys[i][1] = 'None'
                print(f"unlabelled datasets {unique_datasets[i]} with keys {unique_keys[i]}")
            else:
                data_config['dataset']['meta_only_labels'] = False
                print(f"Labelled datasets {unique_datasets[i]} with keys {unique_keys[i]}")
            data_config['dataset']['key_pairs'] = [unique_keys[i]]
            data_config['dataset']['subsets'] = unique_datasets[i]
            data_module = DataModule(data_config)
            data_module.unique_setup()
            data_module.setup()
            datasets.append(deepcopy(data_module))
    else:
        image_key, label_key = wandb.config["dataset"]["key_pairs"][
        0
        ]  # TODO: make this more general
        data_module = DataModule(wandb.config)
        data_module.unique_setup()
        data_module.setup()
        datasets.append(data_module)
        # Get the PyTorch DataLoader objects for the training and validation datasets
    unlabelled_dataloader = None
    for data in datasets:
        if data.config['dataset']['meta_only_labels'] or data.config['dataset']['key_pairs'][0][1] == 'None':
            unlabelled_image_key = data.config['dataset']['key_pairs'][0][0]
            unlabelled_dataloader = data.train_dataloader()
            unlabelled_iter = iter(unlabelled_dataloader)
        else:
            image_key, label_key = data.config["dataset"]["key_pairs"][
                0
            ] 
            train_dataloader = data.train_dataloader()

    dataset_name = "synthetic_sean"
    dataset_path = Path(config['paths']['data']) / 'reformatted_data' / dataset_name
    csv_path = Path(config['paths']['data']) / 'reformatted_data' / f"{dataset_name}.csv"
    if dataset_path.exists():
        print(f"Dataset path {dataset_path} already exists, removing it now")
        shutil.rmtree(dataset_path)
    if csv_path.exists():
        print(f"CSV path {csv_path} already exists, removing it now")
        os.remove(csv_path)

    model = Pix2PixModel(opt)
    model.eval()

    visualizer = Visualizer(opt)

    # create a webpage that summarizes the all results
    web_dir = os.path.join(opt.results_dir, opt.name,
                        '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir,
                        'Experiment = %s, Phase = %s, Epoch = %s' %
                        (opt.name, opt.phase, opt.which_epoch))
    meta_df = pd.DataFrame()
    meta_cols = ['subject_id', 'slice_nr', 'dataset']
    # test
    if wandb.config['data']['to_one_hot']['active']:
        ignored_gt_indices = wandb.config['data']['to_one_hot']['ignore_index']
    else:
        ignored_gt_indices = None
    for i, data_i in enumerate(train_dataloader):
        unlabelled_data = next(unlabelled_iter, None)
        if unlabelled_data is None:
            print(f"Resetting unlabelled dataloader at iteration {i}")
            unlabelled_iter = iter(unlabelled_dataloader)
            unlabelled_data = next(unlabelled_iter, None)
        data_i['lge'] = unlabelled_data[unlabelled_image_key]
        generated = model(data_i, mode='inference')
        subject_df = pd.DataFrame(unlabelled_data['meta_dict'], columns = meta_cols)
        subject_ids = []
        converted_gt = data_i['lge_gt'].argmax(dim=1)
        if ignored_gt_indices is not None:
            for idx in ignored_gt_indices:
                converted_gt += (converted_gt >= idx).int()
        for i in range(generated.shape[0]):
            subject_id = f"{unlabelled_data['meta_dict']['subject_id'][i]}_{unlabelled_data['meta_dict']['slice_nr'][i]}"
            subject_ids.append(subject_id)
            subject_path = dataset_path / subject_id
            os.makedirs(subject_path, exist_ok=True)
            image = nib.Nifti1Image(generated[i,0,:,:].cpu().numpy(), np.eye(4))
            gt = nib.Nifti1Image(converted_gt[i,:,:].cpu().float().numpy(), np.eye(4))
            nib.save(image, subject_path / f"{subject_id}_{image_key}.nii.gz")
            nib.save(gt, subject_path / f"{subject_id}_{label_key}.nii.gz")
        subject_df['SubjectID'] = subject_ids
        meta_df = pd.concat([meta_df, subject_df], ignore_index = True)
    meta_df.to_csv(csv_path)

    # Update the test split file with the synthetic data
    split_path = Path(config['paths']['data']) / 'subject_splits' / config['dataset']['split_file']
    split = yaml.load(open(split_path), Loader=yaml.FullLoader)
    syn_test = sample_from_csv_by_group(csv_path, 1, "dataset", "SubjectID")
    split['test'][dataset_name] = syn_test
    data_to_file(split, split_path)

    webpage.save()
    print("SEAN: Done with test.py!")


if __name__ == '__main__':
    main()