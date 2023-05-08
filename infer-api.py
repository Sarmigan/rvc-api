
#
# Set up
#

# cd AI/RVC
# git clone https://github.com/kalomaze/Retrieval-based-Voice-Conversion-WebUI.git
# cp -R Retrieval-based-Voice-Conversion-WebUI/ RVC
# cd RVC
# mkdir -p pretrained uvr5_weights opt
# cp ../Files/easy-infer.py ./
# cp ../Files/pretrained/* ./pretrained/
# cp ../Files/uvr5_weights/* ./uvr5_weights/
# cp ../Files/hubert_base.pt ./

# python infer-web.py

#
# Manual Infer Encode
#

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

import torch, os

from scipy.io import wavfile
from vc_infer_pipeline import VC
from fairseq import checkpoint_utils
from my_utils import load_audio
from infer_pack.models import SynthesizerTrnMs256NSFsid

config = AttrDict({'device': 'cuda:0', 'is_half': False, 'python_cmd': 'python', 'listen_port': 7865, 'iscolab': False, 'noparallel': False, 'noautoopen': False, 'x_pad': 1, 'x_query': 6, 'x_center': 38, 'x_max': 41})

cpt = torch.load("weights/LeonardCohen.pth", map_location='cpu')
tgt_sr = cpt["config"][-1]
cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]

net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=config.is_half)
del net_g.enc_q
net_g.load_state_dict(cpt["weight"], strict=False)
net_g.eval().to(config.device).float()
vc = VC(tgt_sr, config)

hubert_model = checkpoint_utils.load_model_ensemble_and_task(["hubert_base.pt"], suffix="",)[0][0].to(config.device).float()

audio = load_audio('/home/linux/AI/1_16-moses_sumney-doomed_(Vocals).wav', 16000)

times = [0, 0, 0]

audio_opt = vc.pipeline(
    hubert_model,
    net_g,
    0,
    audio,
    times,
    -24,
    'harvest',
    './logs/LeonardCohen/added_IVF317_Flat_nprobe_1.index',
    # file_big_npy,
    0.76,
    1,
    f0_file=None,
)

wavfile.write( "%s/%s" % ("opt", os.path.basename('audio.wav')), tgt_sr, audio_opt )
os.system("scp opt/audio.wav user@192.168.1.2:~/Desktop;")


