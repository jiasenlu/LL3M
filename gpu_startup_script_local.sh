#!/usr/bin/env bash
set -e

# sudo apt-get -qq install -y libsndfile1-dev
# This script will get ran on the servers

# this locks the python executable down to hopefully stop if from being fiddled with...
screen -d -m python3 -c 'import time; time.sleep(999999999)'

wget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh
bash Anaconda3-2024.02-1-Linux-x86_64.sh -b -p $HOME/Anaconda
source $HOME/anaconda3/bin/activate

# Create venv
conda create --name LL3M python=3.10 -y
conda activate LL3M

pip3 install -U pip
pip3 install -U wheel

pip3 install pyopenssl --upgrade

cd ~/LL3M
python3 -m pip install -e '.[gpu]' --upgrade --force-reinstall --no-cache-dir -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip3 install --upgrade fabric dataclasses tqdm cloudpickle smart_open[gcs] func_timeout aioredis==1.3.1
pip3 install httplib2

# 32 * 1024 ** 3 -> 32 gigabytes
export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=34359738368

# TPU V4 install.
# sudo pip3 uninstall jax jaxlib libtpu-nightly libtpulibtpu-tpuv4 -y
# pip3 install -U pip
# pip3 install jax==0.2.28 jaxlib==0.1.76

# gsutil cp gs://cloud-tpu-tpuvm-v4-artifacts/wheels/libtpu/latest/libtpu_tpuv4-0.1.dev* .
# pip3 install libtpu_tpuv4-0.1.dev*