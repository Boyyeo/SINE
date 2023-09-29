

import argparse, os, sys, glob
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid, save_image
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler




source_model_type = 'castle w/ patch-based fine-tuning' #@param['dog w/o patch-based fine-tuning', 'dog w/ patch-based fine-tuning', 'Girl with a peral earring', 'Monalisa', 'castle w/o patch-based fine-tuning', 'castle w/ patch-based fine-tuning']

model_names = { "dog w/o patch-based fine-tuning":   "dog_wo_patch.ckpt",
                "dog w/ patch-based fine-tuning":    "dog_w_patch.ckpt",
                "Girl with a peral earring":    "girl.ckpt",
                "Monalisa": "monalisa.ckpt",
                "castle w/o patch-based fine-tuning":    "castle_wo_patch",
                "castle w/ patch-based fine-tuning":  "castle_w_patch"}

model_configs = { "dog w/o patch-based fine-tuning":   "./configs/stable-diffusion/v1-inference.yaml",
                "dog w/ patch-based fine-tuning":    "./configs/stable-diffusion/v1-inference_patch.yaml",
                "Girl with a peral earring":    "./configs/stable-diffusion/v1-inference_patch_nearest.yaml",
                "Monalisa": "./configs/stable-diffusion/v1-inference_patch_nearest.yaml",
                "castle w/o patch-based fine-tuning":    "./configs/stable-diffusion/v1-inference.yaml",
                "castle w/ patch-based fine-tuning":  "./configs/stable-diffusion/v1-inference_patch.yaml"}

orig_prompts = { "dog w/o patch-based fine-tuning":   "picture of a sks dog",
                "dog w/ patch-based fine-tuning":    "picture of a sks dog",
                "Girl with a peral earring":    "painting of a sks girl",
                "Monalisa": "painting of a sks lady",
                "castle w/o patch-based fine-tuning":    "picture of a sks castle",
                "castle w/ patch-based fine-tuning":  "picture of a sks castle"}

finetuned_models_dir = 'checkpoints_model'
file_name = model_names[source_model_type]
config_name = model_configs[source_model_type]
fine_tune_prompt = orig_prompts[source_model_type]


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

seed = 42
config = OmegaConf.load('configs/stable-diffusion/v1-inference.yaml')
model = load_model_from_config(config, 'checkpoints_model/sd-v1-4-full-ema.ckpt')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)

sin_config = OmegaConf.load(f"{config_name}")
sin_model = load_model_from_config(config, os.path.join(finetuned_models_dir, file_name+'.ckpt'))
sin_model = sin_model.to(device)

v = 0.7 #@param {type:"slider", min:0, max:1, step:0.05}
K_min = 400 #@param {type:"slider", min:0, max:1000, step:10}
scale = 7.5 #@param {type:"slider", min:1.0, max:50, step:0.5}
ddim_steps = 100
ddim_eta = 0.
H = 512
W = 512

prompt = "a photo of a castle" #@param {'type': 'string'}

extra_config = {
    'cond_beta': v,
    'cond_beta_sin': 1. - v,
    'range_t_max': 1000,
    'range_t_min': K_min
}


from ldm.models.diffusion.guidance_ddim import DDIMSinSampler
sampler = DDIMSinSampler(model, sin_model)

setattr(sampler.model, 'extra_config', extra_config)


batch_size = 1
n_rows = 2
start_code = None
precision_scope = autocast
num_samples = 1

all_samples = list()

with torch.no_grad():
    with precision_scope("cuda"):
        with model.ema_scope():
            with sin_model.ema_scope():
                tic = time.time()
                all_samples = list()
                for n in trange(num_samples, desc="Sampling"):   
                    uc = None
                    if scale != 1.0:
                        uc = model.get_learned_conditioning(batch_size * [""])
                        uc_sin = sin_model.get_learned_conditioning(batch_size * [""])

                    prompts = [prompt] * batch_size
                    prompts_single = [fine_tune_prompt] * batch_size
                    
                    c = model.get_learned_conditioning(prompts)
                    c_sin = sin_model.get_learned_conditioning(prompts_single)
                    
                    shape = [4, H // 8, W // 8]
                    samples_ddim, _ = sampler.sample( S=ddim_steps,
                                                      conditioning=c,
                                                      conditioning_single=c_sin,
                                                      batch_size=batch_size,
                                                      shape=shape,
                                                      verbose=False,
                                                      unconditional_guidance_scale=scale,
                                                      unconditional_conditioning=uc,
                                                      unconditional_conditioning_single=uc_sin,
                                                      eta=ddim_eta,
                                                      x_T=start_code)

                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                    all_samples.append(x_samples_ddim)

                grid = torch.stack(all_samples, 0)
                grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                
                grid = make_grid(grid, nrow=n_rows)

                # to image
                grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                os.makedirs('./output', exist_ok=True)
                Image.fromarray(grid.astype(np.uint8)).save(os.path.join('./output', f'{prompt.replace(" ", "-")}.jpg'))
                #display(Image.open(os.path.join('./output', f'{prompt.replace(" ", "-")}.jpg')))
