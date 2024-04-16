"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import sys
from collections import OrderedDict
from options.train_options import TrainOptions
import data
from util.iter_counter import IterationCounter
from util.visualizer import Visualizer
from trainers.pix2pix_trainer import Pix2PixTrainer
import yaml
from pathlib import Path
from qcardia_data import DataModule
import wandb



# ######## added stuff
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"


def main():
    # parse options
    opt = TrainOptions().parse()

    # print options to help debugging
    print(' '.join(sys.argv))

    # load the dataset
    config_path = Path("/home/bme001/20183502/code/msc-stijn/resources/example-config_original.yaml")

    # The config contains all the model hyperparameters and training settings for the
    # experiment. Additionally, it contains data preprocessing and augmentation
    # settings, paths to data and results, and wandb experiment parameters.
    config = yaml.load(open(config_path), Loader=yaml.FullLoader)
    run = wandb.init(
            project=config["experiment"]["project"],
            name=config["experiment"]["name"],
            config=config,
            save_code=True,
            mode="online",
        )

    # Get the path to the directory where the Weights & Biases run files are stored.
    online_files_path = Path(run.dir)
    print(f"online_files_path: {online_files_path}")
    data_module = DataModule(config)
    data_module.setup()
    dataloader = data_module.train_dataloader()

    # create trainer for our model
    trainer = Pix2PixTrainer(opt)

    # create tool for counting iterations
    iter_counter = IterationCounter(opt, len(dataloader))

    # create tool for visualization
    visualizer = Visualizer(opt)

    for epoch in iter_counter.training_epochs():
        iter_counter.record_epoch_start(epoch)
        for i, data_i in enumerate(dataloader, start=iter_counter.epoch_iter):
            iter_counter.record_one_iteration()

            # Training
            # train generator
            if i % opt.D_steps_per_G == 0:
                trainer.run_generator_one_step(data_i)

            # train discriminator
            trainer.run_discriminator_one_step(data_i)

            # Visualizations
            if iter_counter.needs_printing():
                losses = trainer.get_latest_losses()
                visualizer.print_current_errors(epoch, iter_counter.epoch_iter,
                                                losses, iter_counter.time_per_iter)
                visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)

            if iter_counter.needs_displaying():
                print(f"Displaying images of shape {data_i['lge'].shape}, {data_i['lge_gt'].shape}, {trainer.get_latest_generated().shape}")
                visuals = OrderedDict([('input_label', data_i['lge_gt']),
                                    ('synthesized_image', trainer.get_latest_generated()),
                                    ('real_image', data_i['lge'])])
                visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far)

            if iter_counter.needs_saving():
                print('saving the latest model (epoch %d, total_steps %d)' %
                    (epoch, iter_counter.total_steps_so_far))
                trainer.save('latest')
                iter_counter.record_current_iter()

        trainer.update_learning_rate(epoch)
        iter_counter.record_epoch_end()

        if epoch % opt.save_epoch_freq == 0 or \
        epoch == iter_counter.total_epochs:
            print('saving the model at the end of epoch %d, iters %d' %
                (epoch, iter_counter.total_steps_so_far))
            trainer.save('latest')
            trainer.save(epoch)

    print('Training was successfully finished.')

if __name__ == "__main__":
    main()