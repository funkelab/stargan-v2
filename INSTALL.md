mamba create -n stargan-v2
mamba activate stargan-v2
mamba install python=3.9
mamba install -y pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
