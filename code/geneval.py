import argparse
import json
import os

from accelerate import Accelerator
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid
from torchvision.transforms import ToTensor
from pytorch_lightning import seed_everything
from diffusers import DiffusionPipeline, StableDiffusionPipeline

os.environ["TOKENIZERS_PARALLELISM"] = "true"
from models import Showo, MAGVITv2, get_mask_chedule
from training.prompting_utils import UniversalPrompting, create_attention_mask_predict_next
from training.utils import get_config, flatten_omega_conf, image_transform
from transformers import AutoTokenizer
import torch.nn.functional as F

def get_vq_model_class(model_type):
    if model_type == "magvitv2":
        return MAGVITv2
    else:
        raise ValueError(f"model_type {model_type} not supported.")

torch.set_grad_enabled(False)


def main():
    geneval_file_path = '' # the geneval prompt path
    my_checkpoint = ""  # the checkpoint path
    outdir = ''  # the output path
    showo_model_path = ''
    
    with open(geneval_file_path) as fp:
        metadatas = [json.loads(line) for line in fp]

    config = get_config()
    accelerator = Accelerator(mixed_precision="fp16")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config.model.showo.llm_model_path, padding_side="left")

    uni_prompting = UniversalPrompting(tokenizer, max_text_len=config.dataset.preprocessing.max_seq_length,
                                    special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"),
                                    ignore_id=-100, cond_dropout_prob=config.training.cond_dropout_prob)

    vq_model = get_vq_model_class(config.model.vq_model.type)
    vq_model = vq_model.from_pretrained(config.model.vq_model.vq_model_name)
    vq_model.requires_grad_(False)
    vq_model.eval()
    model = torch.load(showo_model_path, map_location='cpu')
    model.load_state_dict(torch.load(my_checkpoint, map_location=device))
    print("Load weights from checkpoint...")

    model.eval()
    mask_token_id = model.config.mask_token_id
    
    (
        vq_model,
        model
    ) = accelerator.prepare(
        vq_model,
        model
    )

    config.training.guidance_scale = 5
    for index, metadata in enumerate(metadatas):
        seed_everything(42)

        outpath = os.path.join(outdir, f"{index:0>5}")
        os.makedirs(outpath, exist_ok=True)

        prompts = [metadata['prompt']]
        n_rows = 1
        print(f"Prompt ({index: >3}/{len(metadatas)}): '{prompts}'")

        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)
        with open(os.path.join(outpath, "metadata.jsonl"), "w") as fp:
            json.dump(metadata, fp)

        sample_count = 0

        with torch.no_grad():
            all_samples = list()
            for n in trange(4, desc="Sampling"):
                # Generate images
                image_tokens = torch.ones((len(prompts), config.model.showo.num_vq_tokens),
                                      dtype=torch.long, device=device) * mask_token_id
                input_ids, _ = uni_prompting((prompts, image_tokens), 't2i_gen')


                if config.training.guidance_scale > 0:
                    uncond_input_ids, _ = uni_prompting(([''] * len(prompts), image_tokens), 't2i_gen')
                    attention_mask = create_attention_mask_predict_next(torch.cat([input_ids, uncond_input_ids], dim=0),
                                                                        pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                                                                        soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                                                                        eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
                                                                        rm_pad_in_image=True)
                else:
                    attention_mask = create_attention_mask_predict_next(input_ids,
                                                                        pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                                                                        soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                                                                        eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
                                                                        rm_pad_in_image=True)
                    uncond_input_ids = None

                if config.get("mask_schedule", None) is not None:
                    schedule = config.mask_schedule.schedule
                    args = config.mask_schedule.get("params", {})
                    mask_schedule = get_mask_chedule(schedule, **args)
                else:
                    mask_schedule = get_mask_chedule(config.training.get("mask_schedule", "cosine"))

                with torch.no_grad():
                    gen_token_ids, _ = model.t2i_generate(
                        input_ids=input_ids,
                        uncond_input_ids=uncond_input_ids,
                        attention_mask=attention_mask,
                        guidance_scale=config.training.guidance_scale,
                        temperature=config.training.get("generation_temperature", 1.0),
                        timesteps=config.training.generation_timesteps,
                        noise_schedule=mask_schedule,
                        noise_type=config.training.get("noise_type", "mask"),
                        seq_len=config.model.showo.num_vq_tokens,
                        uni_prompting=uni_prompting,
                        config=config,
                    )
                
                gen_token_ids = torch.clamp(gen_token_ids, max=config.model.showo.codebook_size - 1, min=0)
                images = vq_model.decode_code(gen_token_ids)
                images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
                images *= 255.0
                images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
                pil_images = [Image.fromarray(image) for image in images]

                for sample in pil_images:
                    sample.save(os.path.join(sample_path, f"{sample_count:05}.png"))
                    sample_count += 1

        del all_samples

    print("Done.")


if __name__ == "__main__":
    main()

'''
CUDA_VISIBLE_DEVICES=0 accelerate launch geneval.py config=configs/Mask_GRPO_train_512x512.yaml 
'''