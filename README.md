vcmsa
=========
Python library to run vector clustering Multiple Sequence Alignment (vcMSA). 

### About

VcMSA uses protein language models to allow alignment of protein sets with conserved function/structure but poorly conserved sequence.

This is an alpha version of the algorithm code that functions as a proof-of-concept, expect later optimization for speed and efficiency

## Currently out of sync with biorxiv produced alignments (with worse output). To match, use github.com/clairemcwhite/transformer_infrastructure/hf_aligner2.py

[Project Github repo](https://github.com/clairemcwhite/vcmsa)

#### Authors
Claire D. McWhite

Mona Singh

### Citation

CD McWhite, M Singh, "Vector-clustering Multiple Sequence Alignment: Aligning into the twilight zone of protein sequence similarity with protein language models", BioRXiv, 2022

### Inputs

 - Fasta file of protein sequences to be aligned
 - huggingface protein language model
     - Either path to downloaded model, or name of the model e.g. "prot_t5_xl_uniref50"
     - Tested on https://huggingface.co/Rostlab/prot_t5_xl_uniref50 and https://huggingface.co/Rostlab/prot_bert_bfd

### Downloading a language model from Huggingface

- Example for the prot_t5_xl_uniref50 model.
- vcMSA was also tested with prot_bert_bfd model, however output alignments differ from the t5 model

```bash

from transformers import T5Tokenizer, T5Model
tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50")
tokenizer.save_pretrained('prot_t5_xl_uniref50')
model = T5Model.from_pretrained("Rostlab/prot_t5_xl_uniref50")
model.save_pretrained('prot_t5_xl_uniref50')
```


### Make conda environment
- bash vcmsa/make_vcmsa_env.sh 


### Installation
vcmsa can be directly installed with [sudo permission] from pypi

```bash
pip install vcmsa 
```

or 

```bash
easy_install install vcmsa
```
Alternatively, vcmsa can also be installed from source. 

```bash
git clone https://github.com/clairemcwhite/vcmsa.git
cd vcmsa
python setup.py install
#or 
sudo python setup.py install
#or 
python setup.py install --user   
```

As this code is in development, please submit cases where vcmsa fails as a github Issue : [https://github.com/clairemcwhite/vcmsa/issues](https://github.com/clairemcwhite/vcmsa/issues)


### Package usage

```python


### Command line usage

$ vcmsa  


usage: vcmsa [-h] -i FASTA_PATH [-e EMBEDDING_PATH] -o OUT_PATH [-nb] [-sl SEQLIMIT] [-ex] [-fx]
             [-l LAYERS [LAYERS ...]] [-hd HEADS] [-st SEQSIMTHRESH] -m MODEL_NAME [-pca] [-p PADDING] [-l2]

optional arguments:
  -h, --help            show this help message and exit
  -i FASTA_PATH, --in FASTA_PATH
                        Path to fasta
  -e EMBEDDING_PATH, --emb EMBEDDING_PATH
                        Path to embeddings
  -o OUT_PATH, --outfile OUT_PATH
                        Path to outfile
  -bc, --batch_correct
                        If added, do batch correction on sequences
  -sl SEQLIMIT, --seqlimit SEQLIMIT
                        Limit to n sequences. For testing
  -ex, --exclude        Exclude outlier sequences from initial alignment process
  -fx, --fully_exclude  Additionally exclude outlier sequences from final alignment
  -l LAYERS [LAYERS ...], --layers LAYERS [LAYERS ...]
                        Which layers (of 30 in protbert) to select, default = '-16 -15 -14 -13 -12 -11 -10 -9 -8 -7 -6 -5 -4 -3 -2 -1'
  -hd HEADS, --heads HEADS
                        File will one head identifier per line, format layer1_head3
  -st SEQSIMTHRESH, --seqsimthresh SEQSIMTHRESH
                        Similarity threshold for clustering sequences
  -m MODEL_NAME, --model MODEL_NAME
                        Model name or path to local model
  -pca, --pca_plot      If flagged, output 2D pca plot of amino acid clusters
  -p PADDING, --padding PADDING
                        Number of characters of X to add to start and end of sequence (can be important for
                        fragment sequences), default: 10
  -l2, --headnorm       Take L2 normalization of each head

ex.
```bash
fasta=LDLa.vie.20seqs.fasta
layers='-16 -15 -14 -13 -12 -11 -10 -9 -8 -7 -6 -5 -4 -3 -2 -1'
model=prot_t5_xl_uniref50
suffix=.16layer.t5.aln
vcmsa  -i $fasta -o $fasta.$suffix --layers  $layers  -m $model --exclude --pca_plot


```






