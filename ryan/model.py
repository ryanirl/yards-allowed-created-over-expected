"""
The intuition for this model roughly follows the ideas brought up in:

    Multi-agent Trajectory Prediction with Fuzzy Query Attention
    Nitin Kamra, Hao Zhu, Dweep Trivedi, Ming Zhang and Yan Liu
    Advances in Neural Information Processing Systems (NeurIPS), 2020

"""
import torch
import torch.nn as nn
import torch.nn.functional as F 

from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from typing import Optional
from typing import Tuple
from typing import List
from torch import Tensor


class LayerNorm(nn.Module):
    def __init__(self, ndim: int, bias: bool = True, eps: float = 1e-5) -> None:
        """LayerNorm with an optional bias.

        Args:
            ndim (int): The number of dimensions to use for the layer norm. 
            bias (bool): Whether to include the optional bias term.
            eps (float): Small value used for numerical stability.
        
        """
        super(LayerNorm, self).__init__()

        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, self.eps)


class RnnModel(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int = 16, **kwargs) -> None:
        """An RNN block for the NFL Data Bowl 2024 data that can optionally
        accepted batched and padded input.

        Args:
            input_dim (int): The number of features in the input.
            embed_dim (int): The embedding size after the RNN.
            **kwargs: Additional arguments added to the RNN layer at initialization. 

        """
        super(RnnModel, self).__init__()

        self.rnn = nn.GRU(input_dim, embed_dim, batch_first = True, **kwargs)
        
    def forward(
        self, x: Tensor, lengths: Optional[List[int]] = None, hx: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x (torch.Tensor): Tensor of shape (batch, length, n_players, features).
            lengths (Optional[List[int]]): When optionally providing padded batched 
                inputs, you can provide the lengths of the original mismatched inputs
                to batch process them. For more explanation see the following torch
                util functions: `pad_packed_sequence` and `pack_padded_sequence`.
            hx (Optional[torch.Tensor]): The hidden state of the RNN. Can be useful
                for inference-time optimizations. 
        
        Returns:
            torch.Tensor: Tensor of shape (batch, length, n_players, embed_dim).

        """
        b, l, n, _ = x.shape

        embed = x.permute(0, 2, 1, 3).reshape(b * n, l, -1)

        if lengths is not None:
            # This is pretty specific to the NFL dataset. That is, we treat each
            # player as part of the batch, and need to duplicate the lengths for
            # each player. 
            lengths_t = torch.Tensor([l for l in lengths for _ in range(n)])

            # Pack the sequence for optimzized batched processing.
            packed_embed = pack_padded_sequence(embed, lengths_t, batch_first = True)
            packed_embed, hx_i = self.rnn(packed_embed, hx)
            embed, _ = pad_packed_sequence(packed_embed, batch_first = True)
        else:
            embed, hx_i = self.rnn(embed, hx)

        embed = embed.reshape(b, n, l, -1).permute(0, 2, 1, 3)

        return embed, hx_i

    
class InteractionModel(nn.Module):
    def __init__(
        self, input_dim: int, embed_dim: int = 16, n_layers: int = 4, n_heads: int = 1
    ) -> None:
        """This attention-style interaction block is used to model the interaction 
        between the players at an individual frame level. See the `ModelV4` 
        documentation for more intuition. 

        Args:
            input_dim (int): The number of features provided for each player. 
            embed_dim (int): The number of features to project the data to before
                applying the attentions layers. 
            n_layers (int): The number of attention layers to use. 
            n_heads (int): The number of heads each attention layer should use. 

        """
        super(InteractionModel, self).__init__()

        self.n_layers = n_layers

        self.ln_q = LayerNorm(embed_dim, True)
        self.ln_k = LayerNorm(embed_dim, True)
        self.ln_v = LayerNorm(embed_dim, True)
        self.ln_f = LayerNorm(embed_dim, True)

        self.rnn_q = nn.Linear(input_dim, embed_dim)
        self.rnn_k = nn.Linear(input_dim, embed_dim)
        self.rnn_v = nn.Linear(input_dim, embed_dim)

        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, n_heads, batch_first = True)
            for _ in range(n_layers)
        ])

        self.ff_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
            ) for _ in range(n_layers)
        ])

    def forward(self, x: Tensor) -> Tensor:
        x_q = self.ln_q(self.rnn_q(x))
        x_k = self.ln_k(self.rnn_k(x))
        x_v = self.ln_v(self.rnn_v(x))

        b, l, n, _ = x.shape
        x_q = x_q.contiguous().view(b * l, n, -1)
        x_k = x_k.contiguous().view(b * l, n, -1)
        x_v = x_v.contiguous().view(b * l, n, -1)

        for i in range(self.n_layers):
            x, _ = self.attention_layers[i](x_q, x_k, x_v)
            x = x + self.ff_layers[i](x)

            x_q = x
            x_k = x
            x_v = x

        x = x.view(b, l, n, -1)

        return x 


class TrajectoryModel(nn.Module):
    def __init__(self, input_dim: int = 11, embed_dim: int = 32, n_layers: int = 4, n_heads: int = 1) -> None:
        """The trajectory model combines an RNN for encoding the temproal dynamics of each
        playe (naive of player interaction), and an attention-style block for encoding the
        spatial interactions between each player at any point in time (naive of temporal
        information). Also notice that we predict the offsets of position and not the raw
        x/y position itself, and a circularly encoded `dir` feature.

        The intuition for each block is the following:
         - The RNN block is naive of player interaction and is used to learn momentum-like
            features from each player. That is, if a player is going in one direction it is
            likely that the player will continue in that general direction.
         - The attention-style block is used to model the interaction between players. When
            combined with the momentum-like features from the RNN-block, it acts to guide
            (or steer) the momentum in a direction that makes sense spatially. For example,
            if you're a defender in a position to tackle the ball carrier, the
            interaction-block should guide the defender towards the direction of the ball
            carrier, information the RNN-block does not have access to.

        Args:
            input_dim (int): The number of features provided for each player. 
            embed_dim (int): The number of features to project the data to before
                applying the attentions layers. 
            n_layers (int): The number of attention layers to use. 
            n_heads (int): The number of heads each attention layer should use. 

        """
        super(TrajectoryModel, self).__init__()
        
        self.inertia_model = RnnModel(input_dim, embed_dim)
        self.interaction_model = InteractionModel(input_dim, embed_dim, n_layers, n_heads)

        self.ff_0 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ELU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.ff_1 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ELU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.ff_trajectory = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ELU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ELU(),
            nn.Linear(embed_dim, 2),
        )

        self.ff_direction = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ELU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ELU(),
            nn.Linear(embed_dim, 2),
        )

    def forward(
        self, x: Tensor, lengths: Optional[List[int]] = None, hx: Optional[Tensor] = None
    ) -> Tuple[Tuple[Tensor, Tensor], Tensor]:
        """
        Args:
            x (torch.Tensor): The input of shape (batch, seq_len, n_player, 9) where
                the last dimension is organized according like the following:
                [x, y, dir_x, dir_y, dis, x_rel, y_rel, x_diff, y_diff, is_def, is_bc].
                See the `prepare_data` function in `ryan/utils.py` for more information
                about how it's structured. 
            lengths (Optional[List[int]]): When optionally providing padded batched 
                inputs, you can provide the lengths of the original mismatched inputs
                to batch process them. For more explanation see the following torch
                util functions: `pad_packed_sequence` and `pack_padded_sequence`.
            hx (Optional[torch.Tensor]): The hidden state of the RNN. Can be useful
                for inference-time optimizations. 

        Returns:
            trajectory (torch.Tensor): Tensor of shape (batch, seq_len, n_plays, 2)
                that represents the predicted x/y trajectories. 
            direction (torch.Tensor): Tensor of shape (batch, seq_len, n_plays, 2)
                that represents the predicted direction (circularly encoded).
            hx (torch.Tensor): The hidden state of the RNN.

        """
        xy = x[:, :, :, :2].clone().detach() # The xy-values. 

        x_t, hx_i = self.inertia_model(x, lengths, hx)
        x_i = self.interaction_model(x)
        x_o = self.ff_0(x_t) + self.ff_1(x_i)

        trajectory = xy + self.ff_trajectory(x_o) # Predict the offset from the previous xy.
        direction = self.ff_direction(x_o) # Predict the raw direction.

        return (trajectory, direction), hx_i


if __name__ == "__main__":
    x = torch.rand(32, 29, 22, 11)

    model = TrajectoryModel(embed_dim = 32)
    model.load_state_dict(torch.load("models/model_split_0.pt"))
    (trajectory, direction), hx = model(x)

    print("input:", x.shape)
    print("trajectory:", trajectory.shape)
    print("direction:", direction.shape)
    print("hidden state:", hx.shape)

    n_params = sum(p.numel() for p in model.parameters())
    print("Number of params:", n_params)


