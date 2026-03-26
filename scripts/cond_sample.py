"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
from PIL import Image
from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import load_cond_data
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    vein_model_and_diffusion_defaults,
    vein_create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = vein_create_model_and_diffusion(
        **args_to_dict(args, vein_model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()
    
    cond_data = load_cond(args.image_size, args.cond_dir, args.batch_size, args.class_cond)
    

    logger.log("sampling...")
    all_images = []
    all_labels = []
    for i in range(args.num_samples // args.batch_size):
  
        model_kwargs = next(cond_data)
        condd =  model_kwargs['cond']
        path = model_kwargs['path']
        oripath = args.cond_dir 
        newpath = args.outpath  

        del model_kwargs['path']
        model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}
    
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        
        sample = sample_fn(
            model,
            (args.batch_size, 1, args.image_size, args.image_size),
            sams = args.sams,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1) # b,c,h,w -> b,h,w,c
        sample = sample.contiguous()
       
        bat = sample.shape[0]
        image_numpy = sample.cpu().numpy()
        for j in range(image_numpy.shape[0]):
            img_np = image_numpy[j].squeeze().astype(np.uint8)
            img = Image.fromarray(img_np, mode='L')
         
            p = path[j]
            newp = p.replace(oripath, newpath)
            os.makedirs(os.path.dirname(newp), exist_ok=True)
            img.save(newp)
        
        logger.log(f"created {(i+1) * bat} samples")

    logger.log("sampling complete")



def load_cond(image_size, cond_dir, batch_size, class_cond=None):
    data = load_cond_data(
        image_size=image_size,
        cond_dir=cond_dir,
        batch_size=batch_size,
        class_cond=class_cond,
    )
    for c in data:
        yield c

def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=96,
        batch_size=16,
        use_ddim=False,
        model_path="",
        cond_dir="",
        outname="",
        sams=7,
        outpath="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
