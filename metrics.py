#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
from dreamsim import dreamsim
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser

def readImages(renders_dir, gt_dir, preprocess):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append((tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda(), preprocess(render)))
        gts.append((tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda(), preprocess(gt)))
        image_names.append(fname)
    return renders, gts, image_names

def evaluate(model_paths):

   # Initialize DreamSim model and preprocessing function
    cache_dir = os.path.expanduser("~/.cache")
    os.makedirs(cache_dir, exist_ok=True)
    dreamsim_model, preprocess = dreamsim(pretrained=True, cache_dir=cache_dir)
    dreamsim_model = dreamsim_model.cuda()

    full_dict = {}
    per_view_dict = {}
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir / "gt"
                renders_dir = method_dir / "renders"
                renders, gts, image_names = readImages(renders_dir, gt_dir, preprocess)

                ssims = []
                psnrs = []
                lpipss = []
                dreamsims = []  # List to store DreamSim loss values

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    render_tensor, render_preprocessed = renders[idx]
                    gt_tensor, gt_preprocessed = gts[idx]

                    ssims.append(ssim(render_tensor, gt_tensor))
                    psnrs.append(psnr(render_tensor, gt_tensor))
                    lpipss.append(lpips(render_tensor, gt_tensor, net_type='vgg'))
                    dreamsims.append(dreamsim_model(render_preprocessed.cuda(), gt_preprocessed.cuda()).item())  # Compute DreamSim loss

                print("  SSIM     : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("  PSNR     : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("  LPIPS    : {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                print("  DreamSim : {:>12.7f}".format(torch.tensor(dreamsims).mean(), ".5"))
                print("")

                full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                     "PSNR": torch.tensor(psnrs).mean().item(),
                                                     "LPIPS": torch.tensor(lpipss).mean().item(),
                                                     "DreamSim": torch.tensor(dreamsims).mean().item()})
                per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                        "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                        "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                                                        "DreamSim": {name: ds for ds, name in zip(torch.tensor(dreamsims).tolist(), image_names)}})

            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except Exception as e:
            print("Unable to compute metrics for model", scene_dir)
            print("Error:", e)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    evaluate(args.model_paths)
