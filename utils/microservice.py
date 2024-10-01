from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import torch
from model import AttModel
import librosa
import numpy as np
from pathlib import Path
import os
import argparse
import sys
from copy import copy
import importlib
import pandas as pd
from dataset import TestDataset

app = FastAPI()

sys.path.append('./configs')

parser = argparse.ArgumentParser(description="")
parser.add_argument("-C", "--config", help="config filename", default="ait_bird_local")
parser.add_argument("-W", "--weight", help="weight path", default="./weights/ait_bird_local_eca_nfnet_l0/fold_0_model.pt")
parser.add_argument("-D", "--device", help="Model device", default="cuda")
parser_args, _ = parser.parse_known_args(sys.argv)
CFG = copy(importlib.import_module(parser_args.config).cfg)

if parser_args.device == "cuda":
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device:", device)
    else:
        print("Cuda not found, switching to cpu")
        device = torch.device("cpu")
else:
    device = torch.device("cpu")
    print("Using device:", device)
    
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
    
    for inputs in loader:
        row_ids = inputs['row_id']
        inputs.pop('row_id')

        with torch.no_grad():
            output = model(inputs)['logit']

        for row_id_idx, row_id in enumerate(row_ids):
            prediction_dict[str(row_id)]= output[row_id_idx, :].sigmoid().detach().cpu().numpy()
            # prediction_dict[str(row_id)] = F.softmax(output[row_id_idx, :], dim=0).detach().cpu().numpy()
            
    for row_id in list(prediction_dict.keys()):
        logits = np.array(prediction_dict[row_id])
        prediction_dict[row_id] = {}
        classification_dict[row_id] = {}
        for label in range(len(CFG.target_columns)):
            prediction_dict[row_id][CFG.target_columns[label]] = logits[label]
            classification_dict[row_id]['Class'] = CFG.target_columns[np.argmax(logits)]
            classification_dict[row_id]['Score'] = np.max(logits)

    return prediction_dict, classification_dict


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if not file.filename.endswith(('.ogg', '.mp3', '.wav')):
        return JSONResponse(status_code=400, content={"message": "Invalid file type"})

    # Extract just the filename to avoid directory issues
    filename = Path(file.filename).name
    
    temp_folder = 'temp'
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)
        
    audio_path = f'{temp_folder}/{filename}'

    # Save the uploaded file to the temp directory
    contents = await file.read()
    with open(audio_path, 'wb') as f:
        f.write(contents)

    try:
        # Perform predictions
        prediction_dict, classification_dict = prediction_for_clip(audio_path)
        for row_id, predictions in classification_dict.items():
            classification_dict[row_id] = {k: float(v) if isinstance(v, np.float32) else v for k, v in predictions.items()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Clean up temporary file
    os.remove(audio_path)

    return classification_dict


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
