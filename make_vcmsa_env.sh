
conda create --name vcmsa_env -c conda-forge -c pytorch -c bioconda transformers bioconda::mafft pytorch::pytorch=1.12.1 pandas biopython faiss seqeval cudatoolkit=11.3 python-igraph matplotlib  huggingface_hub sentencepiece

# Removing icecream


