# LL3M: Large Language and Multi-Modal Model in Jax / Flax

The goal of this repo is to build a Large Language / Multi-Modal Model and MoE Model that easily trains and finetunes in Jax / Flax.

### Installing on GPU Host
The GPU environment can be installed via [Anaconda](https://www.anaconda.com/products/distribution).

``` shell
conda env create -f scripts/gpu_environment.yml
conda activate LL3M
```

### Installing on Cloud TPU Host
The TPU host VM comes with Python and PIP pre-installed. Run the following
script to set up the TPU host.

``` shell
bash ./tpu_startup_script_local.sh
```

Activate the environment

```
. $HOME/.LL3M/bin/activate
```

## Model

### Large Language Model (LLM)

Currently, the codebase supports LLaMA, Mistral, Phi, OpenLLaMA, and TinyLLaMA models for training and inference. 


## Dataset
### LLM Dataset

The Dolma dataset contains high-quality data from different sources. The OLMo model just concatenated all the tokens without any sampling. 
Here, we use seqio to sample different data based on heuristic factors as below


| Source            | Doc Type      | Bytes     | Percentage    | factor | byte | sample ratio | 
| ------------------| -------       | -------   | --------      | -------| -------- | ------    |
| Common Crawl      | web pages     | 9,022     | 78.46%        | 0.5x | 4,511 | 46.23% | 
| The Stack         | code          | 1,043     | 9.07%         | 2x| 2,086 | 21.37% |
| C4                | web pages     | 790       | 6.87%         | 2x | 1580 | 16.19% |
| Reddit            | social media  | 339       | 2.94%         | 2x | 678 | 6.94% |
| peS2o             | STEM papers   | 268       | 2.33%         | 2x | 536 | 5.49% |
|Project Gutenberg  | books         | 20.4      | 0.17%         | 5x | 204 | 2.10% | 
|Wikipedia, Wikibooks|encyclopedic  | 16.2      | 0.14%         | 5x | 162 | 1.66% |

For more information, please refer to the doc

## Release Plan
- [x] Language Model and Seqio Dataloader for Dolma dataset.
- [x] Multimodal Model that supports LLava, caption, and others. 
- [x] The shaped model combines different variances that can serve as an initial MoE model. 
- [ ] A mixtral type of MoE model can be trained from scratch or existing dense models.
- [ ] DPO and RLHF on LLM, LMM and MoE. 

## Credits
A large portion of the code is borrowed from [EazyLM](https://github.com/young-geng/EasyLM)
