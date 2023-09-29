sudo -s -H apt update
sudo -s -H apt install -y git wget curl htop vim tmux unzip p7zip-full p7zip-rar
sudo -s -H apt install -y python3-pip python3-dev
sudo -s -H pip3 install taming-transformers-rom1504 pydantic git+https://github.com/openai/CLIP.git@main#egg=clip transformers
sudo -s -H pip3 install pytorch-lightning==1.5.9
sudo -s -H pip3 install test-tube
sudo -s -H pip3 install kornia==0.6
sudo -s -H pip3 install Pillow==9.5.0
mkdir checkpoints_model
wget -O checkpoints_model/sd-v1-4-full-ema.ckpt https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4-full-ema.ckpt 
gdown -O checkpoints_model/castle_w_patch.ckpt https://drive.google.com/uc?id=1Cmga--lY-bThvtuLcq4XkI7CVyxnpbIh

