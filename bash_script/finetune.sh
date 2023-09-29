IMG_PATH=test_cases/castle1.jpg
CLS_WRD='castle'
NAME='castle_w_patch_finetune'

python main.py \
    --base configs/stable-diffusion/v1-finetune_patch_picture.yaml \
    -t --actual_resume checkpoints_model/sd-v1-4-full-ema.ckpt \
    -n $NAME --gpus '0,'   --logdir ./logs \
    --data_root $IMG_PATH \
    --reg_data_root $IMG_PATH --class_word $CLS_WRD  