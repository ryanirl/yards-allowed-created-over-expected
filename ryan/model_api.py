import numpy as np
import torch

from typing import Optional
from typing import Tuple
from typing import List
from torch import Tensor

from utils.data import convert_output
from utils.data import prepare_data
from utils.data import combine

from model import TrajectoryModel


class TrajectoryAPI:
    def __init__(self, model: TrajectoryModel, device: str = "cpu") -> None:
        self.model = model.to(device)
        self.model.eval()

        self.device = device

    @torch.no_grad()
    def forward(
        self, x: Tensor, lengths: Optional[List[int]] = None, hx: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """Simple wrapper around the forward call of the model that automatically
        formats the output to be consistant with the input for easy concatenation.

        """
        (trajectory, direction), hx_i = self.model(x, lengths, hx)
        pred = combine(x, trajectory, direction)
        return pred, hx_i

    @torch.no_grad()
    def predict(self, data: np.ndarray, n_steps: int) -> np.ndarray:
        """Predict n-steps into the future given a single batch.

        Args:
            data (np.array): Array of shape (seq_len, n_players, features).
            n_steps (int): The number of time steps to predict into the future.

        Returns:
            np.ndarray: The predicted frames of shape (n_steps, n_players, features).

        """
        x = torch.from_numpy(data).to(self.device).float().unsqueeze(0) 
        _, l, n, c = x.shape # (1, seq_len, n_players, features)

        # The output array to be returned.
        preds = np.zeros((n_steps, n, c), dtype = np.float32)

        # Optimized lopp to the previous version that utilizes the hidden state. 
        hx = None
        for i in range(n_steps):
            x, hx = self.forward(x, hx = hx)

            # Because we have the hidden state `hx`, we don't need to re-evaluate
            # on the rest of the data. 
            x = x[:, -1].unsqueeze(1)
            preds[i] = x[0].cpu().numpy()

        return preds


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    model = TrajectoryModel()
    model.load_state_dict(torch.load("models/model_split_0.pt"))
    model_api = TrajectoryAPI(model)

    game_id = 2022091808
    play_id = 565

    data = np.load("output.npy", allow_pickle = True).item()
    data = data[game_id][play_id]
    l, n, _ = data.shape

    data = data[:l//2]
    inp = prepare_data(data)

    out = model_api.predict(inp, l//2)
    x, y, d, v = convert_output(out)

    print("x", x.shape)
    print("y", y.shape)
    print("d", d.shape)
    print("v", v.shape)

    for i in range(22):
        plt.scatter(
            x = x[:, i],
            y = y[:, i],
            c = "red",
            s = 5
        )

    for i in range(22):
        plt.scatter(
            x = data[:, i, 0],
            y = data[:, i, 1],
            s = 5
        )

    plt.show()

