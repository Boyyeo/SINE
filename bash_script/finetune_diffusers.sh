export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export IMG_PATH="test_cases/castle1.jpg"
export OUTPUT_DIR="output"

CUDA_VISIBLE_DEVICES="2" accelerate launch  --gpu_ids 2 diffusers_train.py   \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --train_text_encoder \
  --img_path=$IMG_PATH \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of a sks castle" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=10000 \
  --checkpointing_steps=1000 \
  --patch_based_training \
  --mixed_precision fp16 \
  --allow_tf32 \
