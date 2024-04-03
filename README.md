# LL3M: Large Language-based and Multi-Modal Model

The goal of this repo is to build a Large Language / Multi-Modal Model and MoE Model that eazily train and finetune in Jax

### Installing on GPU Host
The GPU environment can be installed via [Anaconda](https://www.anaconda.com/products/distribution).

``` shell
conda env create -f scripts/gpu_environment.yml
conda activate LL3M
```

### Installing on Cloud TPU Host
The TPU host VM comes with Python and PIP pre-installed. Simply run the following
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

Currently, the codebase support LLaMA, Mistral, Phi, OpenLLaMA, TinyLLaMA model for training and inference. 


## Dataset
### LLM Dataset

The dolma dataset contains high quality data from different source. The OLMo model just concatenated all the tokens without any sampling. 
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
- [x] Language Model and Seqio Dataloader for dolma dataset.
- [ ] Multimodal Model that support LLava, caption and others. 
- [ ] Mixtral type of MoE model that can train from scratch or existing dense models.
- [ ] Souped model that combine differet variances that can serve as init as MoE model. 
- [ ] DPO and RLHF on LLM, LMM and MoE. 

## Credits
Large portion of the code is borrowed from [EazyLM](https://github.com/young-geng/EasyLM)
