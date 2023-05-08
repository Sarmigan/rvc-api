from flask import Flask

# import os
# from subprocess import Popen
# from random import shuffle

# gt_wavs_dir = "./logs/LeonardCohen/0_gt_wavs"
# co256_dir = "./logs/LeonardCohen/3_feature256"
# f0_dir = "./logs/LeonardCohen/2a_f0"
# f0nsf_dir = "./logs/LeonardCohen/2b-f0nsf"

# names = (
#     set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)])
#     & set([name.split(".")[0] for name in os.listdir(co256_dir)])
#     & set([name.split(".")[0] for name in os.listdir(f0_dir)])
#     & set([name.split(".")[0] for name in os.listdir(f0nsf_dir)])
# )

# opt = []
# for name in names:
#     opt.append(
#         "%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s"
#         % (
#             gt_wavs_dir.replace("\\", "\\\\"),
#             name,
#             co256_dir.replace("\\", "\\\\"),
#             name,
#             f0_dir.replace("\\", "\\\\"),
#             name,
#             f0nsf_dir.replace("\\", "\\\\"),
#             name,
#             0,
#         )
#     )

# for _ in range(2):
#     opt.append("./logs/mute/0_gt_wavs/mute48k.wav|./logs/mute/3_feature256/mute.npy|./logs/mute/2a_f0/mute.wav.npy|./logs/mute/2b-f0nsf/mute.wav.npy|0")

# shuffle(opt)
# with open("./logs/LeonardCohen/filelist.txt", "w") as f:
#     f.write("\n".join(opt))

# cmd = "python train_nsf_sim_cache_sid_load_pretrain.py -e LeonardCohen -sr 48k -f0 1 -bs 8 -g 0 -te 200 -se 20 -pg pretrained/f0G48k.pth -pd pretrained/f0D48k.pth -l 1 -c 0"
# p = Popen(cmd, shell=True, cwd="/home/linux/AI/RVC/RVC")


# cmd = "python train_nsf_sim_cache_sid_load_pretrain.py -e LeonardCohen -sr 48k -f0 1 -bs 4 -g 0 -te 20 -se 5 -pg pretrained/f0G48k.pth -pd pretrained/f0D48k.pth -l 1 -c 0"


# #
# # Create Index
# #

# import faiss
# import numpy as np

# exp_dir = "./logs/LeonardCohen"
# feature_dir = "./logs/LeonardCohen/3_feature256"
# listdir_res = list(os.listdir(feature_dir))

# npys = []
# for name in sorted(listdir_res):
#     phone = np.load("%s/%s" % (feature_dir, name))
#     npys.append(phone)

# big_npy = np.concatenate(npys, 0)
# big_npy_idx = np.arange(big_npy.shape[0])
# np.random.shuffle(big_npy_idx)
# big_npy = big_npy[big_npy_idx]
# np.save("%s/total_fea.npy" % exp_dir, big_npy)

# n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
# index = faiss.index_factory(256, "IVF%s,Flat" % n_ivf)
# index_ivf = faiss.extract_index_ivf(index)
# index_ivf.nprobe = 1
# index.train(big_npy)
# faiss.write_index(index, "%s/trained_IVF%s_Flat_nprobe_%s.index" % (exp_dir, n_ivf, index_ivf.nprobe),)
# index.add(big_npy)
# faiss.write_index(index, "%s/added_IVF%s_Flat_nprobe_%s.index" % (exp_dir, n_ivf, index_ivf.nprobe),)



# #
# # Save Model
# #

# mkdir -p ../Voices/zips/LeonardCohen/
# cp ./logs/LeonardCohen/added_*.index ../Voices/zips/LeonardCohen/
# cp ./logs/LeonardCohen/total_*.npy ../Voices/zips/LeonardCohen/
# cp ./weights/LeonardCohen.pth ../Voices/zips/LeonardCohen/LeonardCohen2333333.pth

# # To save Big File for later updating
# cp ./logs/LeonardCohen/G_2333333.pth ../Voices/zips/LeonardCohen/LeonardCohen_D_2333333.pth
# cp ./logs/LeonardCohen/D_2333333.pth ../Voices/zips/LeonardCohen/LeonardCohen_G_2333333.pth

# zip -r ../Voices/LeonardCohen.zip ../Voices/zips/LeonardCohen
# rm -rf ../Voices/zips/LeonardCohen






