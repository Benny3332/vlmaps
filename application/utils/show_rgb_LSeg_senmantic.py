import os
from pathlib import Path

import cv2
import gdown
import torch
import torchvision.transforms as transforms

from vlmaps.lseg.modules.models.lseg_net import LSegEncNet
from vlmaps.utils.lseg_utils import get_lseg_feat
from vlmaps.utils.matterport3d_categories import mp3dcat
from vlmaps.utils.gml_floor_4_lab import gml4cat
device = ""
clip_feat_dim = 0
def _init_lseg():
    crop_size = 480  # 480
    base_size = 520  # 520
    global device
    global clip_feat_dim
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    lseg_model = LSegEncNet("", arch_option=0, block_depth=0, activation="lrelu", crop_size=crop_size)
    model_state_dict = lseg_model.state_dict()
    checkpoint_dir = Path(__file__).resolve().parents[2]/"vlmaps" / "lseg" / "checkpoints"
    checkpoint_path = checkpoint_dir / "demo_e200.ckpt"
    os.makedirs(checkpoint_dir, exist_ok=True)
    if not checkpoint_path.exists():
        print("Downloading LSeg checkpoint...")
        # the checkpoint is from official LSeg github repo
        # https://github.com/isl-org/lang-seg
        checkpoint_url = "https://drive.google.com/u/0/uc?id=1ayk6NXURI_vIPlym16f_RG3ffxBWHxvb"
        gdown.download(checkpoint_url, output=str(checkpoint_path))

    pretrained_state_dict = torch.load(checkpoint_path, map_location=device)
    pretrained_state_dict = {k.lstrip("net."): v for k, v in pretrained_state_dict["state_dict"].items()}
    model_state_dict.update(pretrained_state_dict)
    lseg_model.load_state_dict(pretrained_state_dict)

    lseg_model.eval()
    lseg_model = lseg_model.to(device)

    norm_mean = [0.5, 0.5, 0.5]
    norm_std = [0.5, 0.5, 0.5]
    lseg_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    clip_feat_dim = lseg_model.out_c
    return lseg_model, lseg_transform, crop_size, base_size, norm_mean, norm_std

if __name__ == "__main__":
    file_path = "/home/benny/data/collect_tran_vlmaps_data/5LpN333mAk7_1/rgb"
    # file_path = "/media/benny/bennyMove/data/collect_tran_vlmaps_data/"
    rgb_file_name = "001094.png"
    rgb_path = Path(file_path) / rgb_file_name
    lseg_model, lseg_transform, crop_size, base_size, norm_mean, norm_std = _init_lseg()
    bgr= cv2.imread(str(rgb_path))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pix_feats = get_lseg_feat(
        lseg_model, rgb, gml4cat[1:-1], lseg_transform, device, crop_size, base_size, norm_mean, norm_std, vis=True
    )