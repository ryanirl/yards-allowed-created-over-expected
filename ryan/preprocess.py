from tqdm.auto import tqdm
import polars as pl
import numpy as np
import logging
import glob
import os

from typing import Union
from typing import Dict
from polars import DataFrame

logger = logging.getLogger(__name__)


def remove_football(df_tracking: DataFrame) -> DataFrame:
    """Removes any rows that are for the football."""
    return df_tracking.filter(pl.col("displayName") != "football")


def add_defensive_attribute(df_tracking: DataFrame, df_plays: DataFrame) -> DataFrame:
    """Joins the plays dataframe on the tracking data, and adds a column
    indicating whether the player is on defense or not (1 for defense and
    0 for offense).

    In my testing, this function takes 5 seconds to run for all of the tracking
    data. This is ~19x speedup to the previous function we had running. 

    """
    play_cols = ["gameId", "playId", "defensiveTeam", "ballCarrierId"]
    df_tracking = df_tracking.join(
        df_plays[play_cols], on = ["gameId", "playId"], how = "left"
    )

    # Boolean array representing whether the player is a defender or not.
    is_def = (df_tracking["club"] == df_tracking["defensiveTeam"])

    # Determine if the player is on defense or not
    df_tracking = df_tracking.with_columns(
        isDef = pl.when(is_def).then(1).otherwise(0)
    )

    # Remove unneeded columns
    df_tracking = df_tracking.drop(["defensiveTeam", "club"])
    
    return df_tracking


def filter_plays_without_start_and_end(df_tracking: DataFrame) -> DataFrame:
    """Filters out any plays that do not have at least both a start and an end
    event.

    Over the previous version, I see roughly a 2.3x speedup.

    """
    # Define the special events for run start and run end
    run_start_events = ["pass_outcome_caught", "handoff", "snap_direct", "run"]
    run_end_events = ["tackle", "out_of_bounds", "fumble", "qb_slide", "qb_sack", "touchdown"]
            
    # Add on the start and end play conditions
    df_tracking = df_tracking.with_columns(
        playStart = df_tracking["event"].is_in(run_start_events),
        playEnd   = df_tracking["event"].is_in(run_end_events)
    )

    # Filter out any plays that do not have both a start and end.
    df_tracking = df_tracking.group_by(["gameId", "playId"]).map_groups(
        lambda group: group.filter(
            (pl.col("playStart") == True).any() &
            (pl.col("playEnd") == True).any()
        )
    )   

    return df_tracking


def remove_frames_not_in_play(group: DataFrame) -> DataFrame:
    """This function will remove any frames that are not within the time that
    the ball is in play. 

    Below is a rough intuition behind the vectorization. 
     1. start = [0,  0,  1,  0,  0,  0,  0,  0]
     2. end   = [0,  0,  0,  0,  0,  0,  1,  0]
     3. roll  = [0,  0,  0,  0,  0,  0,  0,  1] # roll(end, 1) 
     4. sub   = [0,  0,  1,  0,  0,  0,  0, -1] # start - roll(end, 1)
     5. sum   = [0,  0,  1,  1,  1,  1,  1,  0] # cumsum(start - roll(end, 1))

    Within a group_by for gameId and playId, this is meant to give us the active
    playId's for a single player (inner). These playId's can then be used to 
    filter the grouping as a whole.

    See the note about wrapping for step 3 in the code.

    Takes ~20 seconds to run for all of the tracking data. 

    Example usage:

     >>> df_tracking: pl.DataFrame = ...
     >>> df_tracking = df_tracking.group_by(["gameId", "playId"]).map_groups(
     >>>     remove_frames_not_in_play
     >>> )

    """
    # We do an initial filter by displayName because nflId has null values for
    # the football, which Polars does not like.
    inner = group.filter(group["displayName"] == group["displayName"][0])
    if not ((inner["playStart"].sum() == 1) & (inner["playEnd"].sum() == 1)):
        return group.filter(False)

    inner = inner.with_columns(playStart = inner["playStart"].fill_null(False))
    inner = inner.with_columns(playEnd   = inner["playEnd"  ].fill_null(False))

    start = inner["playStart"].to_numpy().astype(int)
    end   = inner["playEnd"].to_numpy().astype(int)

    # We want to include the last frame where it ends!
    end = np.roll(end, shift = 1)
    end[0] = 0 # Handle the case in which it's the last frame

    filt = np.cumsum(start - end).astype(bool)
    frame_ids = inner.filter(filt)["frameId"]

    group = group.filter(
        group["frameId"].is_in(frame_ids)
    )
    return group


def reverse_deg(deg: Union[int, float]) -> Union[int, float]:
    if deg < 180:
        return deg + 180

    return deg - 180


def flip_plays_left_to_right(df_tracking: DataFrame) -> DataFrame:
    inds = (df_tracking["playDirection"] == "left")
    
    df_tracking = df_tracking.with_columns(
        x = pl.when(inds).then(120.0 - pl.col("x")).otherwise(pl.col("x")),
        o = pl.when(inds).then(pl.col("o").map_elements(reverse_deg)).otherwise(pl.col("o")),
        dir = pl.when(inds).then(pl.col("dir").map_elements(reverse_deg)).otherwise(pl.col("dir"))
    )

    return df_tracking


def sort(df_tracking: DataFrame) -> DataFrame:
    return df_tracking.sort(["gameId", "playId"])


def _add_in_play_col(group: DataFrame) -> DataFrame:
    #group = group.with_columns(nflId = inner["playStart"].fill_null(False))
    inner = group.filter(group["displayName"] == group["displayName"][0])

    # Now find run the tests
    keep = (
        (inner["playStart"].sum() == 1) & 
        (inner["playEnd"  ].sum() == 1)
    )
    
    if not keep:
        return group.filter(False)

    inner = inner.with_columns(playStart = inner["playStart"].fill_null(False))
    inner = inner.with_columns(playEnd   = inner["playEnd"  ].fill_null(False))

    start = inner["playStart"].to_numpy().astype(int)
    end   = inner["playEnd"].to_numpy().astype(int)

    # We want to include the last frame where it ends!
    end = np.roll(end, shift = 1)
    end[0] = 0 # Handle the case in which it's the last frame

    filt = np.cumsum(start - end).astype(bool)
    frame_ids = inner.filter(filt)["frameId"]
    
    group = group.with_columns(
        inPlay = pl.when(group["frameId"].is_in(frame_ids)).then(1).otherwise(0)
    )
    return group


def add_in_play_col(df_tracking: DataFrame) -> DataFrame:
    """Adds a binary column for whether the current frame is 'in-play'. That is,
    between the time before the ball carrier get's the ball and hasn't yet been
    tackled or etc.

    """
    df_tracking = df_tracking.with_columns(inPlay = 0)
    df_tracking = df_tracking.group_by(["gameId", "playId"]).map_groups(_add_in_play_col)

    return df_tracking


def add_ball_carrier(df_tracking: DataFrame) -> DataFrame:
    # Add on a boolean for the ball carrier
    df_tracking = df_tracking.with_columns(
        nflId = df_tracking["nflId"].fill_null(0)
    )
    df_tracking = df_tracking.with_columns(
        ball_carrier = (df_tracking["nflId"] == df_tracking["ballCarrierId"])
    )
    # Have ball carrier only be ball carrier when the play is inPlay
    inds = (pl.col("ball_carrier") & (pl.col("inPlay") > 0.5))
    df_tracking = df_tracking.with_columns(
        ball_carrier = pl.when(inds).then(True).otherwise(False)
    )

    return df_tracking


def write_data(output_file: str, df_tracking: DataFrame) -> None:
    """Loads the data into a dictionary that can be indexed like the
    following:

        data[game_id][play_id]

    The indexing returns an np.ndarray of shape (seq_len, n_players, 9).

    The data writes to an npy file in a pickled format and can be loaded
    back into memory with the following:

        data = np.load(output_file, allow_pickle = True).item()

    """
    # Features important for training/inference. Everything else we drop.
    features = ["x", "y", "s", "a", "dis", "o", "dir", "isDef", "ball_carrier"]
    n_features = len(features)

    data: Dict[str, Dict[str, np.ndarray]] = {}
    game_ids = df_tracking["gameId"].unique(maintain_order = True)
    for game_id in tqdm(game_ids):
        df_game = df_tracking.filter(df_tracking["gameId"] == game_id)

        data[game_id] = {}

        for play_id in df_game["playId"].unique(maintain_order = True):
            df_play = df_game.filter(df_game["playId"] == play_id)
            frame_ids = sorted(df_play["frameId"].unique(maintain_order = True))

            play = np.zeros((len(frame_ids), 22, n_features))
            for i, frame_id in enumerate(frame_ids):
                group = df_play.filter(df_play["frameId"] == frame_id)
                group = group.sort(["nflId"])

                play[i] = group[features].to_numpy()

            data[game_id][play_id] = play

    np.save(output_file, data, allow_pickle = True)


def preprocess(data_dir: str, output_file: str = "data.npz", safe: bool = True) -> None:
    """Code for preprocessing the raw NFL Big Data Bowl 2024 Kaggle data.
    
    Requires about 6 Gb of free RAM. 

    """
    tracking_files = sorted(glob.glob(os.path.join(data_dir, "tracking_week_*.csv")))

    # Do some validation of the input.
    if safe:
        if os.path.exists(output_file):
            raise FileExistsError(f"The output file already exists: '{output_file}'")    

        for file_path in tracking_files: 
            if not os.path.exists(file_path):
                raise FileExistsError(
                    f"The following tracking file does not exist: '{file_path}'"
                )

    # Maybe 8-9x faster loading over Pandas
    logger.info("[1  / 10]: Loading dataframes into memory")
    df_plays    = pl.read_csv(os.path.join(data_dir, "plays.csv"),   null_values = ["NA"])
    df_tracking = pl.concat(
        [pl.read_csv(file, null_values = ["NA"]) for file in tracking_files]
    )

    # The pipeline
    logger.info("[2  / 10]: Removing frames without a football")
    df_tracking = remove_football(df_tracking)

    logger.info("[3  / 10]: Filtering plays without both a start and end")
    df_tracking = filter_plays_without_start_and_end(df_tracking)

    logger.info("[4  / 10]: Flipping left plays -> right plays")
    df_tracking = flip_plays_left_to_right(df_tracking) # Largely for training.

    logger.info("[5  / 10]: Sorting tracking data")
    df_tracking = sort(df_tracking) # Groupby's mess up sorting in Polars

    logger.info("[6  / 10]: Adding defensive attributes")
    df_tracking = add_defensive_attribute(df_tracking, df_plays)

    logger.info("[7  / 10]: Adding in-play column")
    df_tracking = add_in_play_col(df_tracking)

    logger.info("[8  / 10]: Adding ball carrier column")
    df_tracking = add_ball_carrier(df_tracking)

    logger.info("[9  / 10]: Sorting tracking data")
    df_tracking = sort(df_tracking) 

    logger.info(f"[10 / 10]: Writing data to '{output_file}'.")
    write_data(output_file, df_tracking)


if __name__ == "__main__":
    import argparse
    from utils import setup_logger

    name = "preprocessing"
    desc = "Script for preprocessing the raw NFL Big Data Bowl 2024 Kaggle data."

    parser = argparse.ArgumentParser(prog = name, description = desc)
    parser.add_argument(
        "-i", "--data_dir", type = str, default = "../data/", metavar = "",
        help = "Location to the folder containing all of the Kaggle data."
    )
    parser.add_argument(
        "-o", "--output_file", type = str, default = "output.npz", metavar = "",
        help = "Output file for the data, must have the .npy ending."
    )
    args = parser.parse_args()

    setup_logger()
    preprocess(args.data_dir, args.output_file)


