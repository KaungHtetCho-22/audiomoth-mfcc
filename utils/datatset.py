import torch
import pandas as pd
import numpy as np
import librosa
import os

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 df: pd.DataFrame, 
                 clip: np.ndarray,
                 cfg=None,
                ):
        
        self.df = df
        self.clip = clip
        self.sr = cfg.sample_rate
        self.config = cfg

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):

        sample = self.df.loc[idx, :]
        row_id = sample.row_id

        end_seconds = int(sample.seconds)
        start_seconds = int(end_seconds - 5)
        
        wave = self.clip[self.sr * start_seconds : self.sr * end_seconds].astype(np.float32)
            
        return {
            "row_id": row_id,
            "wave": wave,
            "rating": torch.ones(1),
            "loss_target": torch.ones(1),
            "embedding": torch.rand(264).unsqueeze(0),
        }


