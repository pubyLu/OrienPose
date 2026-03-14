from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from torch.utils.data import DataLoader

from src.models.model import PoseConditional
from src.models.u_net.denoising_diffusion_pytorch.u_net import UNet

from src.models.encoder.AutoencoderKL import VAE_StableDiffusion
import torch

def batch_to_device(batch: Any, device: torch.device | str, non_blocking: bool = True):

    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=non_blocking)
    if isinstance(batch, np.ndarray):
        # 保持数值类型，避免对象数组
        if batch.dtype == np.object_:
            raise TypeError("numpy 数组含有 object 类型，无法安全转成 Tensor")
        t = torch.from_numpy(batch)
        return t.to(device, non_blocking=non_blocking)
    if isinstance(batch, Mapping):
        return {k: batch_to_device(v, device, non_blocking) for k, v in batch.items()}
    if isinstance(batch, tuple):
        return tuple(batch_to_device(x, device, non_blocking) for x in batch)
    if isinstance(batch, Sequence) and not isinstance(batch, (str, bytes)):
        return [batch_to_device(x, device, non_blocking) for x in batch]
    # 其它类型（如标量、字符串）保持原样
    return batch


def test(model, dataloader, device):
    model.eval()
    results ={}
    with torch.no_grad():
        for j, batch in enumerate(dataloader):
            print(f"please waiting for batch{j}...")
            batch = batch_to_device(batch, device)
            names = batch['name']
            nearest_idx = model.myTest_step(batch)
            for i in range(len(names)):
                results[names[i]] = nearest_idx[i]

    return results


from pathlib import Path
from src.dataloader.shapenet import ShapeNet


if __name__ == '__main__':
    vae = VAE_StableDiffusion(latent_dim=4,
                              name="VAE",
                              pretrained_path="/project_root/pretrained/stable-diffusion-v1-5_vae.pth").cuda()

    unet = UNet(u_net_dim=192,
                rot_representation_dim=6,
                encoder=vae,
                pose_mlp_name="single_layer",
                )

    checkpoints = "./results/orienpose.ckpt" # cvpr version
    save_dir = Path("path/to/test_data/")
    model = PoseConditional.load_from_checkpoint(
        checkpoints,
        u_net=unet,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(device)

    txt_file = f"test.txt"
    # load dataset -- Reader needs to revise the code of ShapeNet_dataset_class to make your data work.
    '''
    one batch has:
        batch['query']
        batch['ref']
        batch['name'] # be used to save result for specific sample
        batch['template_relRs']
    '''
    test_dataset = ShapeNet(root_dir="path/to/data",
                                txtFile=txt_file,
                                img_size=256)
    print("test dataset lens:", len(test_dataset))
    test_dataloser = DataLoader(test_dataset, batch_size=32, shuffle=False)
    test(model, test_dataloser, device)


