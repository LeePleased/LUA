An implementation of Lexical Unit Analysis (LUA) for sequence segmentation tasks (e.g., Chinese POS Tagging). Note that this is not an officially supported Tencent product.

# Preparation

Two steps. Firstly, reformulate the chunking data sets and move them into a new folder named "dataset". The folder contains {train, dev, test}.json. 
Each JSON file is a list of dicts. See the following NER case:
```
[ 
 {
  "sentence": "['Somerset', '83', 'and', '174', '(', 'P.', 'Simmons']",
  "labeled entities": "[(0, 0, 'ORG'), (1, 1, 'O'), (2, 2, 'O'), (3, 3, 'O'), (4, 4, 'O'), (5, 6, 'PER')]",
 },
 {
  "sentence": "['Leicestershire', '22', 'points', ',', 'Somerset', '4', '.']",
  "labeled entities": "[(0, 0, 'ORG'), (1, 1, 'O'), (2, 2, 'O'), (3, 3, 'O'), (4, 4, 'ORG'), (5, 5, 'O'), (0, 0, 'O')]",
 }
]
```

Secondly, pretrained LM (i.e., [BERT](https://www.aclweb.org/anthology/N19-1423/)) and [evaluation script](https://www.clips.uantwerpen.be/conll2000/chunking/conlleval.txt). 
Create another directory, "resource", with the following arrangement:
- resource
    - pretrained_lm
        - model.pt
        - vocab.txt
    - conlleval.pl

For Chinese tasks, the source to construct "pretrained_lm" is bert-base-chinese.

## Training and Test
```
CUDA_VISIBLE_DEVICES=0 python main.py -dd dataset -sd dump -rd resource
```
