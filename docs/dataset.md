
1. Download dolma dataset from instructions [here](https://huggingface.co/datasets/allenai/dolma#download).

Generate `file.txt` from the script:
```
python create_data/dolma/get_dolma.py
```

Download the dataset using aria2c
```
aria2c --input-file files.txt --header 'Authorization: Bearer YOUR_HF_HUB_ACCESS_TOKEN'
```

2. Preprocess dolma dataset to tfrecord for data loading. 

We use the [dolma](https://huggingface.co/datasets/allenai/dolma) v1.6 dataset to train the LLM from scratch.  

```
wget https://huggingface.co/datasets/allenai/dolma/edit/main/urls/v1_6.txt
```

Download the dataset using aria2c
```
aria2c --input-file files.txt -s 10
```

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
