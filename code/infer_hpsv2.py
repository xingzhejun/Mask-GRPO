import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"
from PIL import Image
from accelerate import Accelerator
from tqdm import tqdm
import numpy as np
import torch
# import wandb
from models import Showo, MAGVITv2, get_mask_chedule
from training.prompting_utils import UniversalPrompting, create_attention_mask_predict_next
from training.utils import get_config, flatten_omega_conf, image_transform
from transformers import AutoTokenizer
import torch.nn.functional as F

import csv
import shutil

def get_vq_model_class(model_type):
    if model_type == "magvitv2":
        return MAGVITv2
    else:
        raise ValueError(f"model_type {model_type} not supported.")

if __name__ == '__main__':
    prompts_file = '' # please infer to the hpsvd2 test_data
    my_checkpoint = ""  # the checkpoint path
    outdir = ''  # the output path
    showo_model_path = ''

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

    prompts = []
    with open(prompts_file, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader) # header
        for row in reader:
            if row:  # 确保行不是空的
                prompts.append(row[0])
    print('test_data长度为:', len(prompts))

    config.training.batch_size = config.batch_size
    config.training.guidance_scale = config.guidance_scale
    config.training.generation_timesteps = config.generation_timesteps
    # load from users passed arguments

    if config.mode == 't2i':
        save_id =0 
        config.training.guidance_scale = 5
        for step in tqdm(range(0, len(prompts), config.training.batch_size)):
            prompt = [prompts[step]]
            image_tokens = torch.ones((len(prompt), config.model.showo.num_vq_tokens),
                                      dtype=torch.long, device=device) * mask_token_id

            input_ids, _ = uni_prompting((prompt, image_tokens), 't2i_gen')

            if config.training.guidance_scale > 0:
                uncond_input_ids, _ = uni_prompting(([''] * len(prompt), image_tokens), 't2i_gen')
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
            safe_prompt = "".join(c if c.isalnum() or c in " _-" else "_" for c in prompt[0])[:200]
            img_save_path = f"{save_id}_{safe_prompt}.jpg"
            pil_images[0].save(outdir + img_save_path)
            print(f"Image saved to {img_save_path}")
            save_id = save_id + 1
