import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import torch
from tqdm.auto import tqdm
import argparse
import sys
from copy import copy
import importlib
from dataset import TestDataset
from model import AttModel
import torch.nn.functional as F

warnings.filterwarnings("ignore")

sys.path.append('./configs')

parser = argparse.ArgumentParser(description="")
parser.add_argument("-C", "--config", help="config filename", default="ait_bird_local")
parser.add_argument("-W", "--weight", help="weight path", default="./weights/ait_bird_local_eca_nfnet_l0/fold_0_model.pt")
parser.add_argument("-A", "--audio", help="audio file path", default="./data/soundscape_29201.ogg")
parser.add_argument("-E", "--export", help="export folder path", default="./exports/")
parser_args, _ = parser.parse_known_args(sys.argv)
CFG = copy(importlib.import_module(parser_args.config).cfg)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
state_dict = torch.load(parser_args.weight, map_location=device)['state_dict']

model = AttModel(
    backbone=CFG.backbone,
    num_class=CFG.num_classes,
    infer_period=5,
    cfg=CFG,
    training=False,
    device=device
)

model.load_state_dict(state_dict)
model = model.to(device)
model.logmelspec_extractor = model.logmelspec_extractor.to(device)

def prediction_for_clip(audio_path):
    
    prediction_dict = {}
    classification_dict = {}
    
    clip, _ = librosa.load(audio_path, sr=32000)
    
    # Get the duration of the clip in seconds and calculate intervals
    duration = librosa.get_duration(y=clip, sr=32000)
    seconds = list(range(5, int(duration), 5))  # Ensure it covers the whole audio length
    
    filename = Path(audio_path).stem
    
    # Generate row ids for each segment
    row_ids = [filename + f"_{second}" for second in seconds]

    test_df = pd.DataFrame({
        "row_id": row_ids,
        "seconds": seconds,
    })
    
    dataset = TestDataset(
        df=test_df, 
        clip=clip,
        cfg=CFG,
    )
        
    loader = torch.utils.data.DataLoader(
        dataset,
        **CFG.loader_params['valid']
    )
    
    for i, inputs in enumerate(tqdm(loader)):
        
        row_ids = inputs['row_id']
        inputs.pop('row_id')

        with torch.no_grad():
            output = model(inputs)['logit']

        for row_id_idx, row_id in enumerate(row_ids):
            prediction_dict[str(row_id)]= output[row_id_idx, :].sigmoid().detach().cpu().numpy()
            # prediction_dict[str(row_id)] = F.softmax(output[row_id_idx, :], dim=0).detach().numpy()
            
    for row_id in list(prediction_dict.keys()):
        logits = np.array(prediction_dict[row_id])
        prediction_dict[row_id] = {}
        classification_dict[row_id] = {}
        for label in range(len(CFG.target_columns)):
            prediction_dict[row_id][CFG.target_columns[label]] = logits[label]
            classification_dict[row_id]['Class'] = CFG.target_columns[np.argmax(logits)]
            classification_dict[row_id]['Score'] = np.max(logits)

    return prediction_dict, classification_dict

prediction_dict, classification_dict = prediction_for_clip(parser_args.audio)

os.makedirs(parser_args.export, exist_ok=True)

logit_df = pd.DataFrame.from_dict(prediction_dict, "index").rename_axis("row_id").reset_index()
logit_export_path = os.path.join(parser_args.export, Path(parser_args.audio).stem+"_logits.csv")
logit_df.to_csv(logit_export_path, index=False)

classification_df = pd.DataFrame.from_dict(classification_dict, "index").rename_axis("row_id").reset_index()
classification_export_path = os.path.join(parser_args.export, Path(parser_args.audio).stem+"_classification.csv")
classification_df.to_csv(classification_export_path, index=False)

print(f"Done! Exported predictions to {logit_export_path} and {classification_export_path}")