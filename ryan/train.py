import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import seaborn as sns
import numpy as np
import logging
import json
import time
import glob
import sys
import os

from torch import Tensor
from typing import Optional
from typing import Union
from typing import Tuple
from typing import List
from typing import Dict

import torch.nn.functional as F
import torch.optim as optim 
import torch.nn as nn
import torch

from utils.logger import setup_logger
from utils.dataloader import MultiEpochsDataLoader
from utils.data import prepare_data
from utils.common import save_model
from nfl_dataset import NflDataset
from nfl_dataset import collate_fn
from model import TrajectoryModel

logger = logging.getLogger(__name__)


def load_data(data_path: str, split: float = 0.7) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    df = np.load(data_path, allow_pickle = True).item()

    # The dataset is super small so just load it all in.
    data = []  
    for game_id in tqdm(df.keys()):
        for play_id in df[game_id].keys():
            play = np.nan_to_num(df[game_id][play_id])
            play = prepare_data(play)
            
            data.append(play)
        
    start = int(len(data) * split) 
    training_data = data[:start]
    testing_data  = data[start:]

    logger.info(f"Training dataset size: {len(training_data)}")
    logger.info(f"Testing dataset size: {len(testing_data)}")

    return training_data, testing_data


def model_step(
    model: TrajectoryModel, 
    optimizer: optim.Optimizer, 
    loss_fn: nn.modules.loss._Loss, 
    x: Tensor, 
    y: Tensor, 
    seq_lens: List[int], 
    scale: Union[int, float] = 1_000,
    training: bool = True,
    device: str = "cuda"
) -> Tuple[float, float]:
    """
    """
    (traj, direction), _ = model(x, seq_lens)
    
    dir_loss = torch.zeros(1, device = device)
    traj_loss = torch.zeros(1, device = device)
    for i in range(len(seq_lens)):  
        dir_loss = dir_loss + loss_fn(direction[i, :seq_lens[i]-1], y[i, 1:seq_lens[i], :, 2:4]) / scale
        traj_loss = traj_loss + loss_fn(traj[i, :seq_lens[i]-1],  y[i, 1:seq_lens[i], :, :2]) 

    dir_loss = dir_loss / len(seq_lens)
    traj_loss = traj_loss / len(seq_lens)
    loss = dir_loss + traj_loss
    
    if training:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none = True)

    return traj_loss.item(), dir_loss.item() * scale


def training_loop(config: Dict) -> None:
    model = TrajectoryModel(
        input_dim = config["input_dim"],
        embed_dim = config["embed_dim"],
        n_layers = config["n_layers"],
        n_heads = config["n_heads"]
    ).to(config["device"])

    n_params = sum(p.numel() for p in model.parameters())

    logger.info(model)
    logger.info(f"Parameter count: {n_params}")
    logger.info(json.dumps(config, indent = 4))

    logger.info("Loading the data into memory")
    training_data, testing_data = load_data(config["data_path"], config["split"])
    train_dataset = NflDataset(data = training_data)
    train_dataloader = MultiEpochsDataLoader(
        train_dataset, 
        batch_size = config["batch_size"],
        collate_fn = collate_fn,
        pin_memory = True,
        drop_last = True,
        shuffle = True,
        num_workers = 1
    )
    test_dataset = NflDataset(data = testing_data)
    test_dataloader = MultiEpochsDataLoader(
        test_dataset, 
        batch_size = config["batch_size"],
        collate_fn = collate_fn,
        pin_memory = True,
        drop_last = True,
        shuffle = True,
        num_workers = 1
    )

    optimizer = optim.Adam(model.parameters(), lr = config["lr"])
    loss_function = nn.MSELoss()

    epochs = config["epochs"]
    iters = len(train_dataloader)
    delimiter = " | "

    testing_direction_loss_values = [] 
    testing_trajectory_loss_values = [] 
    training_direction_loss_values = [] 
    training_trajectory_loss_values = [] 

    logger.info("---------- Starting Training ----------")

    total_time = time.time()
    for epoch in range(1, epochs + 1):
        epoch_time = time.time()
        
        epoch_testing_direction_loss_values = [] 
        epoch_testing_trajectory_loss_values = [] 
        epoch_training_direction_loss_values = [] 
        epoch_training_trajectory_loss_values = [] 
        
        model.train(True)
        for step, (x, y, seq_lens) in enumerate(train_dataloader, 1):
            step_time = time.time()
            
            batch, seq_len, n_players, n_features = x.shape
            batch, seq_len, n_players, n_features = y.shape
            
            x = x.to(config["device"], non_blocking = True).float()
            y = y.to(config["device"], non_blocking = True).float()

            traj_loss, dir_loss = model_step(
                model, optimizer, loss_function, x, y, seq_lens, device = config["device"]
            )
            
            epoch_training_direction_loss_values.append(dir_loss)
            epoch_training_trajectory_loss_values.append(traj_loss)

            if step % config["log_every"] == 0:
                curr_time = time.time()
                logger.info(delimiter.join([
                    f"Epoch: [{epoch:>{len(str(epochs))}}/{epochs}]", 
                    f"Step: [{step:>{len(str(iters))}}/{iters}]",
                    f"Step Time: {curr_time - step_time:6.4f}",
                    f"Total Time: {curr_time - total_time:8.4f}",
                    f"Traj Loss: {traj_loss:.10f}",
                    f"Dir Loss: {dir_loss * 1_000:.10f}",
                ]))

        model.eval()
        logger.info("---------- Starting Testing ----------")
        with torch.no_grad():
            for step, (x, y, seq_lens) in enumerate(test_dataloader, 1):
                step_time = time.time()

                batch, seq_len, n_players, n_features = x.shape
                batch, seq_len, n_players, n_features = y.shape

                x = x.to(config["device"], non_blocking = True).float()
                y = y.to(config["device"], non_blocking = True).float()

                traj_loss, dir_loss = model_step(
                    model, optimizer, loss_function, x, y, seq_lens, training = False, device = config["device"]
                )

                epoch_testing_direction_loss_values.append(dir_loss)
                epoch_testing_trajectory_loss_values.append(traj_loss)

        logger.info(
            f"Trajectory Loss: {np.mean(epoch_testing_trajectory_loss_values)} | "
            f"Direction Loss: {np.mean(epoch_testing_direction_loss_values)}"
        )
        logger.info("---------- Finished Testing ----------")
        
        testing_direction_loss_values.append(epoch_testing_direction_loss_values)
        testing_trajectory_loss_values.append(epoch_testing_trajectory_loss_values)
        training_direction_loss_values.append(epoch_training_direction_loss_values)
        training_trajectory_loss_values.append(epoch_training_trajectory_loss_values)

    logger.info("---------- Finished Training ----------")

    logger.info(f"Saving model to {config['output_file']}")
    save_model(config["output_file"], model)

    log_plot(
        testing_values = testing_trajectory_loss_values,
        training_values = training_trajectory_loss_values,
        title = "Trajectory Loss",
        save_as = "trajectory_loss.png"
    )
    log_plot(
        testing_values = testing_direction_loss_values,
        training_values = training_direction_loss_values,
        title = "Direction Loss",
        save_as = "direction_loss.png"
    )


def log_plot(testing_values, training_values, title, save_as):
    fig, ax = plt.subplots(1, 1, figsize = (5, 3))
    ax.plot(np.log(np.mean(testing_values, axis = 1)), label = "Testing Set", linewidth = 3)
    ax.plot(np.log(np.mean(training_values, axis = 1)), label = "Training Set", linewidth = 3)
    ax.set_title(title)
    ax.set_ylabel("MSE Loss (Log)")
    ax.set_xlabel("Epoch")
    plt.legend(title = "Data Split", loc = "center left", bbox_to_anchor = (0.65, 0.85), frameon = False)
    sns.despine()
    plt.savefig(save_as, facecolor = "white", bbox_inches = "tight")


def parse_args():
    import argparse

    name = "training"
    desc = "The main training function for the trajectory forcasting model."

    parser = argparse.ArgumentParser(
        prog = name, description = desc, formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-i", "--data-path", type = str, default = "output.npy", metavar = "",
        help = "where the processed '.npy' file is from 'preprocess.py'. "
    )
    parser.add_argument(
        "-o", "--output-file", type = str, default = "model.pt", metavar = "",
        help = "where to save the trained model file too."
    )
    parser.add_argument(
        "--split", type = float, default = 0.7, metavar = "",
        help = "ratio of train-to-test split."
    )
    parser.add_argument(
        "--epochs", type = int, default = 50, metavar = "", 
        help = "the number of epochs to train for."
    )
    parser.add_argument(
        "--batch_size", type = int, default = 32, metavar = "", 
        help = "the batch size to using during trainng. Each item is a play."
    )
    parser.add_argument(
        "--lr", type = float, default = 3e-4, metavar = "",
        help = "the learning rate for Adam."
    )
    parser.add_argument(
        "--device", type = str, default = "cuda", metavar = "",
        help = "the device to use. Common options: 'cuda' or 'cpu'."
    )
    parser.add_argument(
        "--log-every", type = int, default = 25, metavar = "",
        help = "after how many steps during training to log the current loss."
    )
    parser.add_argument(
        "--input-dim", type = int, default = 11, metavar = "",
        help = "the number of features given to the model. Probably DON'T chagne this."
    )
    parser.add_argument(
        "--embed-dim", type = int, default = 32, metavar = "",
        help = "the embedding dimension to use for the model"
    )
    parser.add_argument(
        "--n-layers", type = int, default = 4, metavar = "",
        help = "the number of attention layers to use for the interaction model."
    )
    parser.add_argument(
        "--n-heads", type = int, default = 1, metavar = "",
        help = "the number of attention heads to use for the interaction model."
    )

    return parser.parse_args()


def main():
    setup_logger()
    args = vars(parse_args())
    training_loop(args)


if __name__ == "__main__":
    main()



