from infer_web import train1key
from typing import Annotated
from multiprocessing import cpu_count
from fastapi import FastAPI, Depends, Query
import uvicorn

from config import Config
config = Config()

class TrainParams:
    def __init__(self,
                 exp_dir1: Query(str, description="Folder name (Must have no spaces and be unique)", example="custom_aditya"),
                 sr2: Query(str, description="Sample rate (32k, 40k or 48k)", example="32k"),
                 if_f0_3: Query(str, description="Extract pitch information (yes or no)", example="yes"),
                 trainset_dir4: Query(str, description="Directory of training data (Must have no spaces)", example="/app/data/custom_aditya"),
                 spk_id5: Query(int, description="Identifier for the speaker", example="0"),
                 np7: Query(str, description=f"Number of cpu threads for pitch extraction (0-{cpu_count()})", example=f"{cpu_count()}"),
                 f0method8: Query(str, description="Method for extracting pitch information (pm, harvest or dio)", example="harvest"),
                 save_epoch10: Query(str, description="Save every nth epoch (0-50)", example="5"),
                 total_epoch11: Query(str, description="Number of epochs (0-1000)", example="150"),
                 batch_size12: Query(str, description="Batch size (1-40)", example="1"),
                 if_save_latest13: Query(str, description="Save only latest checkpoint (yes or no)", example="yes"),
                 pretrained_G14: Query(str, description="Directory of pretrained G (Must have no spaces)", example="/app/pretrained/f0G40k.pth"),
                 pretrained_D15: Query(str, description="Directory of pretrained D (Must have no spaces)", example="/app/pretrained/f0D40k.pth"),
                 gpus16: Query(str, description="Card numbers used (Seperated by -, for example 0-1-2)", example="0"),
                 if_cache_gpu17: Query(str, description="Cache dataset into GPU memory (yes or no)", example="no")):
        self.exp_dir1 = exp_dir1
        self.sr2 = sr2
        self.if_f0_3 = if_f0_3
        self.trainset_dir4 = trainset_dir4
        self.spk_id5 = spk_id5
        self.np7 = np7
        self.f0method8 = f0method8
        self.save_epoch10 = save_epoch10
        self.total_epoch11 = total_epoch11
        self.batch_size12 = batch_size12
        self.if_save_latest13 = if_save_latest13
        self.pretrained_G14 = pretrained_G14
        self.pretrained_D15 = pretrained_D15
        self.gpus16 = gpus16
        self.if_cache_gpu17 = if_cache_gpu17

app = FastAPI()

@app.get("/")
async def root():
    return {"status": "200"}

@app.get("/train")
async def train(commons: Annotated[TrainParams, Depends(TrainParams)]):
    train1key(
        commons.exp_dir1,
        commons.sr2,
        commons.if_f0_3,
        commons.trainset_dir4,
        commons.spk_id5,
        commons.np7,
        commons.f0method8,
        commons.save_epoch10,
        commons.total_epoch11,
        commons.batch_size12,
        commons.if_save_latest13,
        commons.pretrained_G14,
        commons.pretrained_D15,
        commons.gpus16,
        commons.if_cache_gpu17
    )
    return

if __name__ == "__main__":
    uvicorn.run("train-api:app", host="0.0.0.0", port=config.listen_port, log_level="info")