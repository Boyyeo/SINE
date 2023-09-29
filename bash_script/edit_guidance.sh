LOG_DIR='validate'
CHECKPOINT_PATH='checkpoints_model/last.ckpt'
GPU_ID=3


python stable_txt2img_guidance.py --ddim_eta 0.0 --n_iter 1 \
    --scale 10 --ddim_steps 100 \
    --sin_config configs/stable-diffusion/v1-inference_patch.yaml \
    --sin_ckpt $CHECKPOINT_PATH \
    --prompt "a watercolor painting of a castle[SEP]picture of a sks castle" \
    --ckpt checkpoints_model/sd-v1-4-full-ema.ckpt\
    --cond_beta 0.7 \
    --range_t_min 1000 --range_t_max 1000 --single_guidance \
    --skip_save --H 768 --W 768 --n_samples 1 \
    --outdir $LOG_DIR --gpu $GPU_ID

python stable_txt2img_guidance.py --ddim_eta 0.0 --n_iter 1 \
    --scale 10 --ddim_steps 100 \
    --sin_config configs/stable-diffusion/v1-inference_patch.yaml \
    --sin_ckpt $CHECKPOINT_PATH \
    --prompt "a watercolor painting of a castle[SEP]picture of a sks castle" \
    --ckpt checkpoints_model/sd-v1-4-full-ema.ckpt\
    --cond_beta 0.7 \
    --range_t_min 800 --range_t_max 1000 --single_guidance \
    --skip_save --H 768 --W 768 --n_samples 1 \
    --outdir $LOG_DIR --gpu $GPU_ID

python stable_txt2img_guidance.py --ddim_eta 0.0 --n_iter 1 \
    --scale 10 --ddim_steps 100 \
    --sin_config configs/stable-diffusion/v1-inference_patch.yaml \
    --sin_ckpt $CHECKPOINT_PATH \
    --prompt "a watercolor painting of a castle[SEP]picture of a sks castle" \
    --ckpt checkpoints_model/sd-v1-4-full-ema.ckpt\
    --cond_beta 0.7 \
    --range_t_min 600 --range_t_max 1000 --single_guidance \
    --skip_save --H 768 --W 768 --n_samples 1 \
    --outdir $LOG_DIR --gpu $GPU_ID

python stable_txt2img_guidance.py --ddim_eta 0.0 --n_iter 1 \
    --scale 10 --ddim_steps 100 \
    --sin_config configs/stable-diffusion/v1-inference_patch.yaml \
    --sin_ckpt $CHECKPOINT_PATH \
    --prompt "a watercolor painting of a castle[SEP]picture of a sks castle" \
    --ckpt checkpoints_model/sd-v1-4-full-ema.ckpt\
    --cond_beta 0.7 \
    --range_t_min 400 --range_t_max 1000 --single_guidance \
    --skip_save --H 768 --W 768 --n_samples 1 \
    --outdir $LOG_DIR --gpu $GPU_ID

python stable_txt2img_guidance.py --ddim_eta 0.0 --n_iter 1 \
    --scale 10 --ddim_steps 100 \
    --sin_config configs/stable-diffusion/v1-inference_patch.yaml \
    --sin_ckpt $CHECKPOINT_PATH \
    --prompt "a watercolor painting of a castle[SEP]picture of a sks castle" \
    --ckpt checkpoints_model/sd-v1-4-full-ema.ckpt\
    --cond_beta 0.7 \
    --range_t_min 200 --range_t_max 1000 --single_guidance \
    --skip_save --H 768 --W 768 --n_samples 1 \
    --outdir $LOG_DIR --gpu $GPU_ID

python stable_txt2img_guidance.py --ddim_eta 0.0 --n_iter 1 \
    --scale 10 --ddim_steps 100 \
    --sin_config configs/stable-diffusion/v1-inference_patch.yaml \
    --sin_ckpt $CHECKPOINT_PATH \
    --prompt "a watercolor painting of a castle[SEP]picture of a sks castle" \
    --ckpt checkpoints_model/sd-v1-4-full-ema.ckpt\
    --cond_beta 0.7 \
    --range_t_min 0 --range_t_max 1000 --single_guidance \
    --skip_save --H 768 --W 768 --n_samples 1 \
    --outdir $LOG_DIR --gpu $GPU_ID