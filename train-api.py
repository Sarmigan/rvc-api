import os
from subprocess import Popen
from random import shuffle
from fastapi import FastAPI
import uvicorn
from i18n import I18nAuto
from config import Config
config = Config()

now_dir = os.getcwd()
i18n = I18nAuto()

def click_train(
    exp_dir1,
    sr2,
    if_f0_3,
    spk_id5,
    save_epoch10,
    total_epoch11,
    batch_size12,
    if_save_latest13,
    pretrained_G14,
    pretrained_D15,
    gpus16,
    if_cache_gpu17,
):
    # 生成filelist (generate filelist)
    exp_dir = "%s/logs/%s" % (now_dir, exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)
    gt_wavs_dir = "%s/0_gt_wavs" % (exp_dir)
    co256_dir = "%s/3_feature256" % (exp_dir)
    if if_f0_3 == i18n("是"):
        f0_dir = "%s/2a_f0" % (exp_dir)
        f0nsf_dir = "%s/2b-f0nsf" % (exp_dir)
        names = (
            set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)])
            & set([name.split(".")[0] for name in os.listdir(co256_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0nsf_dir)])
        )
    else:
        names = set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)]) & set(
            [name.split(".")[0] for name in os.listdir(co256_dir)]
        )
    opt = []
    for name in names:
        if if_f0_3 == i18n("是"):
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    co256_dir.replace("\\", "\\\\"),
                    name,
                    f0_dir.replace("\\", "\\\\"),
                    name,
                    f0nsf_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
        else:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    co256_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
    if if_f0_3 == i18n("是"):
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature256/mute.npy|%s/logs/mute/2a_f0/mute.wav.npy|%s/logs/mute/2b-f0nsf/mute.wav.npy|%s"
                % (now_dir, sr2, now_dir, now_dir, now_dir, spk_id5)
            )
    else:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature256/mute.npy|%s"
                % (now_dir, sr2, now_dir, spk_id5)
            )
    shuffle(opt)
    with open("%s/filelist.txt" % exp_dir, "w") as f:
        f.write("\n".join(opt))
    print("write filelist done")

    # 生成config#无需生成config
    # cmd = python_cmd + " train_nsf_sim_cache_sid_load_pretrain.py -e mi-test -sr 40k -f0 1 -bs 4 -g 0 -te 10 -se 5 -pg pretrained/f0G40k.pth -pd pretrained/f0D40k.pth -l 1 -c 0"
    print("use gpus:", gpus16)
    if gpus16:
        cmd = (
            config.python_cmd
            + " train_nsf_sim_cache_sid_load_pretrain.py -e %s -sr %s -f0 %s -bs %s -g %s -te %s -se %s -pg %s -pd %s -l %s -c %s"
            % (
                exp_dir1,
                sr2,
                1 if if_f0_3 == i18n("是") else 0,
                batch_size12,
                gpus16,
                total_epoch11,
                save_epoch10,
                pretrained_G14,
                pretrained_D15,
                1 if if_save_latest13 == i18n("是") else 0,
                1 if if_cache_gpu17 == i18n("是") else 0,
            )
        )
    else:
        cmd = (
            config.python_cmd
            + " train_nsf_sim_cache_sid_load_pretrain.py -e %s -sr %s -f0 %s -bs %s -te %s -se %s -pg %s -pd %s -l %s -c %s"
            % (
                exp_dir1,
                sr2,
                1 if if_f0_3 == i18n("是") else 0,
                batch_size12,
                total_epoch11,
                save_epoch10,
                pretrained_G14,
                pretrained_D15,
                1 if if_save_latest13 == i18n("是") else 0,
                1 if if_cache_gpu17 == i18n("是") else 0,
            )
        )
    print(cmd)
    p = Popen(cmd, shell=True, cwd=now_dir)
    p.wait()
    return

app = FastAPI()

@app.get("/")
async def root():
    return {"status": "200"}

@app.get("/train")
async def train(exp_dir1: str,
                sr2: int,
                if_f0_3: str,
                spk_id5: int,
                save_epoch10: int,
                total_epoch11: int,
                batch_size12: int,
                if_save_latest13: str,
                pretrained_G14: str,
                pretrained_D15: str,
                gpus16: bool,
                if_cache_gpu17: str
):
    click_train(exp_dir1=exp_dir1,
                sr2=sr2,
                if_f0_3=if_f0_3,
                spk_id5=spk_id5,
                save_epoch10=save_epoch10,
                total_epoch11=total_epoch11,
                batch_size12=batch_size12,
                if_save_latest13=if_save_latest13,
                pretrained_G14=pretrained_G14,
                pretrained_D15=pretrained_D15,
                gpus16=gpus16,
                if_cache_gpu17=if_cache_gpu17)

if __name__ == "__main__":
    uvicorn.run("train-api:app", host="0.0.0.0", port=config.listen_port, log_level="info")