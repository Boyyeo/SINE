pip install -r requirements.txt
mkdir checkpoints_model
wget -O checkpoints_model/sd-v1-4-full-ema.ckpt https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4-full-ema.ckpt 
gdown -O checkpoints_model/castle_w_patch.ckpt https://drive.google.com/uc?id=1Cmga--lY-bThvtuLcq4XkI7CVyxnpbIh
