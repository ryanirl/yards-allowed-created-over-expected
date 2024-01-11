import torch
from torch.utils.data import Dataset

from torch import Tensor
from numpy import ndarray
from typing import Tuple
from typing import List


class NflDataset(Dataset):
    def __init__(self, data: List[ndarray]) -> None:
        self.data = data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, i: int) -> Tuple[ndarray, ndarray]:
        x = self.data[i] # (seq_len, n_players, 11)
        y = x[:, :, :4]  # (seq_len, n_players, 11)
        return x, y
    

def collate_fn(
    batch: List[Tuple[ndarray, ndarray]]
) -> Tuple[Tensor, Tensor, List[int]]:
    """Custom collate function for batched processing of mixed length 
    sequences (that we have in the NFL dataset). It can be used like the
    following: 

    ```
    train_dataset = NflDataset(x = training_data)
    train_dataloader = MultiEpochsDataLoader(
        train_dataset, 
        batch_size = BATCH_SIZE,
        collate_fn = collate_fn,
        pin_memory = True,
        drop_last = True,
        shuffle = True,
        num_workers = 1
    )
    ```

    """
    x = [torch.Tensor(b[0]).float() for b in batch]
    y = [torch.Tensor(b[1]).float() for b in batch]

    # Sorting the array will give us access to various optmizations that we can
    # take advantage of when using the `pad_sequence`, `pack_padded_sequence`,
    # and `pad_packed_sequence` functions in PyTorch.
    sorted_x = sorted(x, key = lambda x: x.shape[0], reverse = True)
    sorted_y = sorted(y, key = lambda x: x.shape[0], reverse = True)

    # Pad them. This is how we *initially* get the sequences to match length.
    sequences_x = torch.nn.utils.rnn.pad_sequence(sorted_x, batch_first = True)
    sequences_y = torch.nn.utils.rnn.pad_sequence(sorted_y, batch_first = True)

    lengths = [len(a) for a in sorted_x]

    return sequences_x, sequences_y, lengths



