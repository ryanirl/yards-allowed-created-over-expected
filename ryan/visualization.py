from tqdm.auto import tqdm
from PIL import Image
import polars as pl
import numpy as np
import imageio
import os
import io

from typing import Optional
from typing import Union
from typing import List

import plotly.graph_objects as go

from utils.football_field import get_line_of_scrimmage
from utils.football_field import get_field_number_bottom
from utils.football_field import get_field_number_top
from utils.football_field import get_layout


def get_title(df_play: pl.DataFrame, game_id: int, play_id: int) -> str:
    description = df_play["playDescription"][0]
    quarter = df_play["quarter"][0]
    clock = df_play["gameClock"][0]

    words = description.split(" ")
    if (len(words) > 15) and (len(description) > 115):
        description = " ".join(words[0:16]) + "<br>" + " ".join(words[16:])

    title = f"GameId: {game_id}, PlayId: {play_id}<br>{clock} {quarter}Q"
    title = title + ("<br>" * 19) + description 

    return title


def write_animation(fig: go.Figure, animation: List[go.Frame], layout: go.Layout, save_as: str, fps: int) -> None:
    extension = os.path.splitext(save_as)[1].lower()
    
    if extension == ".html":
        fig.write_html(save_as)

    elif extension == ".json":
        fig.write_json(save_as)

    elif extension in [".mp4", ".gif"]:
        image_arr = []
        for frame in tqdm(animation, desc = "Writing video"):
            fig_img = go.Figure(frame["data"], layout = layout).to_image(format = "png")
            fig_buf = io.BytesIO(fig_img)
            fig_arr = np.array(Image.open(fig_buf))
            fig_arr = fig_arr[0:550, 10:] # To remove excess white space. 
            fig_arr[500:] = fig_arr[0, 0]
            image_arr.append(fig_arr)

        imageio.mimwrite(save_as, image_arr, duration = 1000.0 * (1.0 / fps))

    else:
        print(
            f"Extension '{extension}' not found. Options are '.html', '.json', "
            f"'.mp4', or '.gif'. Skipping saving the animation."
        )


def animate_play(
    df_tracking: pl.DataFrame, 
    df_plays: pl.DataFrame, 
    game_id: int = 2022091808, 
    play_id: int = 565,
    predictions: Optional[np.ndarray] = None, 
    scale: Union[int, float] = 10,
    save_as: Optional[str] = None,
    fps: int = 10
) -> go.Figure:
    """
    """
    # Process the input dataframes.
    df_tracking = df_tracking.filter(
        (pl.col("gameId") == game_id) &
        (pl.col("playId") == play_id) &
        (pl.col("displayName") != "football")
    )
    df_play = df_plays.filter(
        (df_plays["gameId"] == game_id) &
        (df_plays["playId"] == play_id) 
    )

    frame_ids = sorted(df_tracking["frameId"].unique())
    n_frames = len(frame_ids)

    # If the play is to the left, then flip the trajectory predictions which
    # are normalized to the right... Asuming predictions are provided.
    if (df_tracking["playDirection"][0] == "left") and (predictions is not None):
        predictions[:, :, :, 0] = 120 - predictions[:, :, :, 0] 

    # Get the field visualization information.
    title = get_title(df_play, game_id, play_id)
    line_of_scrimmage = get_line_of_scrimmage(df_play["absoluteYardlineNumber"][0])
    field_number_bottom = get_field_number_bottom()
    field_number_top = get_field_number_top()
    layout = get_layout(title, n_frames, scale)

    animation = []
    for frame_id in frame_ids:
        frame = []

        # Add field information
        frame.append(field_number_bottom)
        frame.append(field_number_top)
        frame.append(line_of_scrimmage)

        # Plot the predicted and previous positions of each player. 
        if predictions is not None:
            df_prev = df_tracking.filter(df_tracking["frameId"] < frame_id)
            prev_points = go.Scatter(
                x = df_prev["x"].to_numpy(), 
                y = df_prev["y"].to_numpy(), 
                showlegend = False,
                mode = "markers",
                marker = {"color": "black", "opacity": 0.7, "size": 2}
            )
            pred_points = go.Scatter(
                x = predictions[frame_id-1, :, :, 0].reshape(-1),
                y = predictions[frame_id-1, :, :, 1].reshape(-1),
                showlegend = False,
                mode = "markers",
                marker = {"color": "black", "opacity": 0.7, "size": 2}
            )

            frame.append(prev_points)
            frame.append(pred_points)

        # Plot current point of each team colored.
        for team in sorted(df_tracking["club"].unique()):
            df_curr = df_tracking.filter(
                (df_tracking["club"] == team) & 
                (df_tracking["frameId"] == frame_id)
            )

            # Color different based on the team.
            color = "red" if team == df_play["defensiveTeam"][0] else "blue"

            hover_text_array = []
            for nfl_id, display_name in zip(df_curr["nflId"], df_curr["displayName"]):
                hover_text_array.append(
                    f"nflId:{nfl_id}<br>displayName:{display_name}"
                )

            curr_points = go.Scatter(
                x = df_curr["x"], 
                y = df_curr["y"],
                mode = "markers",
                marker_color = color,
                name = team,
                hovertext = hover_text_array,
                hoverinfo = "text"
            )

            frame.append(curr_points)

        # Add the frame to the animation list.
        animation.append(go.Frame(data = frame, name = str(frame_id)))

    fig = go.Figure(
        data = animation[0]["data"],
        layout = layout,
        frames = animation[1:]
    )

    if save_as is not None:
        write_animation(fig, animation, layout, save_as, fps)

    return fig


def parse_args():
    import argparse

    name = "visualization"
    desc = (
        "Script for plotting an animation of a play, and optionally of the "
        "performance of the trajectory forcasting model."
    )

    parser = argparse.ArgumentParser(
        prog = name, description = desc, formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "-i", "--data-dir", default = "../data/", type = str, metavar = "",
        help = "the path to the folder containing the Kaggle data."
    )
    parser.add_argument(
        "-o", "--save-as", default = None, type = str, metavar = "",
        help = "optional param to save the figure as an html, json, gif, or mp4."
    )
    parser.add_argument(
        "--fps", default = 10, type = int, metavar = "",
        help = "if saving as a video, this is the fps to save as."
    )
    parser.add_argument(
        "--game-id", default = 2022091808, type = int, metavar = "",
        help = "the game id to plot"
    )
    parser.add_argument(
        "--play-id", default = 565, type = int, metavar = "",
        help = "the play id to plot"
    )
    parser.add_argument(
        "--model-path", default = "./models/model_split_0.pt", type = str, metavar = "",
        help =  "the path to the file containing the model state dict."
    )
    parser.add_argument(
        "--model-data-path", default = "./output.npy", type = str, metavar = "",
        help = "the path to the preprocessed data."
    )
    parser.add_argument(
        "--n-pred", default = 10, type = int, metavar = "",
        help = "How many points to predict into the future with the trajectory model."
    )

    return parser.parse_args()


def main():
    import torch
    import os

    from model import TrajectoryModel
    from model_api import TrajectoryAPI
    from model_api import prepare_data
    from model_api import convert_output

    args = parse_args()

    df_tracking = pl.read_csv(os.path.join(args.data_dir, "tracking_week_2.csv"), null_values = ["NA"])
    df_plays = pl.read_csv(os.path.join(args.data_dir, "plays.csv"), null_values = ["NA"])

    play = np.load(args.model_data_path, allow_pickle = True).item()
    play = play[args.game_id][args.play_id]
    play = prepare_data(play)

    model = TrajectoryModel()
    model.load_state_dict(torch.load(args.model_path))

    model_api = TrajectoryAPI(model)

    # Predict on every single frame n_steps into the future.
    l, n, c = play.shape
    predictions = np.zeros((l, args.n_pred, n, 4)) 
    for t in range(1, l):
        out = model_api.predict(play[:t], args.n_pred)
        predictions[t] = np.stack(convert_output(out)).transpose(1, 2, 0)

    fig = animate_play(
        df_tracking, 
        df_plays, 
        args.game_id, 
        args.play_id, 
        predictions,
        save_as = args.save_as,
        fps = args.fps
    )
    fig.show()


if __name__ == "__main__":
    main()


