mkdir -p datasets/LLaVA-Pretrain
wget https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/images.zip?download=true -O datasets/LLaVA-Pretrain/images.zip
unzip datasets/LLaVA-Pretrain/images.zip
rm datasets/LLaVA-Pretrain/images.zip

wget https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/blip_laion_cc_sbu_558k.json?download=true -O datasets/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json
