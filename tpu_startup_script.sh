#!/usr/bin/env bash
set -e

# sudo apt-get -qq install -y libsndfile1-dev
# This script will get ran on the servers

# this locks the python executable down to hopefully stop if from being fiddled with...
screen -d -m python3 -c 'import time; time.sleep(999999999)'

# install the latest 
# sudo apt-get install -y -qq software-properties-common
# sudo add-apt-repository -y ppa:deadsnakes/ppa
# sudo apt-get install -y -qq python3.10-full python3.10-dev

# Create venv
python3.10 -m venv $HOME/.LL3M
. $HOME/.LL3M/bin/activate

pip3 install -U pip
pip3 install -U wheel

pip3 install pyopenssl --upgrade

cd ~
python3 -m pip install -e '.[tpu]' --upgrade --force-reinstall --no-cache-dir -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
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