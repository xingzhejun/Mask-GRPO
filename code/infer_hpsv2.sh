export PATH=''
export HF_HOME=''

CUDA_VISIBLE_DEVICES=4 accelerate launch infer_hpsv2.py config=configs/Mask_GRPO_train_512x512.yaml batch_size=1 guidance_scale=5 generation_timesteps=50 mode='t2i'