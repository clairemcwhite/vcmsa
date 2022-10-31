import pickle
import argparse
import faiss
import numpy as np

'''
Do dimension reduction on a pkl of embeddings (either sequence or aa)
Reduces the final dimension in array, saves pca transform arrays.
Ex. 4096 -> 100

Works directly on output of hf_embed.py
 
A pickle of saved pca matrix + bias can also be applied to new embeddings with this script

#### Example use cases
1. Train a new PCA on all aa_embeddings and apply to all embeddings, saving pickle of pca matrix for later
$  python transformer_infrastructure/embedding_pca.py -p test.pkl -o test_dimreduced.pkl -t 100 -e aa_embeddings -om test.pca.matrixbias.pkl

2. Train a new PCA on subset of 100000 sentence_embeddings and apply that pca to all embeddings
$  python transformer_infrastructure/embedding_pca.py -p test.pkl -o test_dimreduced.pkl -t 100 -e sequence_embeddings -s 100000

3. Apply an old PCA training  (saved with -om on previous run)
$  python transformer_infrastructure/embedding_pca.py -p test.pkl -o test_dimreduced.pkl -e sequence_embeddings -im test.pca.matrixbias.pkl

4. Just train a PCA, don't apply it
$  python transformer_infrastructure/embedding_pca.py -p test.pkl -e sequence_embeddings -om test.pca.matrixbias.pkl


#### Plotting principal components
    with open('pkl_out', "rb") as f:
      
         cache_emb = pickle.load(f)
         tr = cache_emb['pcamatrix']p

         plt.scatter(tr[:,0], tr[:, 1],
         c=I_list, edgecolor='none', alpha=0.5,
         cmap=plt.cm.get_cmap('jet', k))
         plt.xlabel('component 1')
        plt.ylabel('component 2')
    plt.colorbar()

    plt.plot(tr[:, 0], tr[:, 1])

    plt.savefig("pca.png")


##### Minimal anaconda environment 
conda create --name embed_pca -c conda-forge numpy faiss pickle argparse


Claire D. McWhite
7/8/2021
'''

def get_pca_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inpickle", dest = "pkl_in", type = str, required = True,
                        help="input .pkl filename")
    parser.add_argument("-o", "--outpickle", dest = "pkl_out", type = str, required = False,
                        help="output .pkl filename for dimension reduced matrix. If not present, pca will be trained, but not applied.")
    parser.add_argument("-e", "--embedding_name", dest = "embedding_name", type = str, default = 'aa_embeddings',
                        help="Name of embedding to reduce. Usually aa_embeddings or sequence_embeddings")
    parser.add_argument("-t", "--target_dim", dest = "target_dim", type = int, default = 100,
                        help="Number of dimensions to reduce to")
    parser.add_argument("-om", "--outpickle_pcamatrix", dest = "pkl_pca_out", type = str, required = False,
                        help="Save necessary arrays to apply pca to new embeddings (y = A.T @ x + b")
    parser.add_argument("-im", "--inpickle_pcamatrix", dest = "pkl_pca_in", type = str, required = False,
                        help="If present, will apply a previous pca transform to all embeddings instead of training a new one")
    parser.add_argument("-s", "--subsample_n", dest = "max_train_sample_size", type = int, required = False,
                        help="If present, will only use n vectors to train pca")

    args = parser.parse_args()    
    return(args)


def reshape_3d_to_2d(arr_3d):
    '''
    Faiss pca takes 2d array, but aa_embeddings are 3d
    Convert to 2d
    
    Takes: array of shape (numseqs x seqlens x n)
    Returns: array of shape (numseqs *  seqlens x n)
    '''
    arr_2d = np.reshape(arr_3d, (arr_3d.shape[0]*arr_3d.shape[1], arr_3d.shape[2]))
    print("Reshaping array from 3d to 2d")
    print("prev_shape: {}".format(arr_3d.shape))
    print("new_shape: {}".format(arr_2d.shape))
    return(arr_2d)

def reshape_2d_to_3d(arr_2d, seqlen, d2):
    '''
    After PCA dimension reduction, return aa_embeddings to original shape
    Convert to 3d
    
    Takes: array of shape (numseqs * seqlens x n)
    Returns: array of shape (numseqs x  seqlens x n)
    '''
    arr_3d = np.reshape(arr_2d, (-1, seqlen, d2))
    print("Reshaping array from 2d to 3d")
    print("prev_shape: {}".format(arr_2d.shape))
    print("new_shape: {}".format(arr_3d.shape))
    return(arr_3d)


def train_pca(hidden_states, target = 100, max_train_sample_size = None):
    '''
    Takes 2d array of hidden states. 
    Will train PCA to reduce second dimension from n to target

    If max_train_sample_size is provided, a random sample of n from hidden states will be used to train. 

    Returns: pcamatrix and bias
    '''
    d1 = hidden_states.shape[1]

    pca = faiss.PCAMatrix(d1, target)
    if max_train_sample_size and hidden_states.shape[0] > max_train_sample_size: 
        rnd_indices = np.random.choice(len(hidden_states), size=max_train_sample_size)
        hidden_states_train = hidden_states[rnd_indices]
        pca.train(np.array(hidden_states_train))

    else:
        pca.train(np.array(hidden_states))

    
    bias = faiss.vector_to_array(pca.b)
    pcamatrix = faiss.vector_to_array(pca.A).reshape(pca.d_out, pca.d_in)
    return(pcamatrix, bias)
       


def apply_pca(hidden_states, pcamatrix, bias):
    '''
    Applies a pcamatrix + bias to hidden states to reduce their 2nd dimension
    
    Returns: reduced hidden_states (2d)
    '''

    reduced = np.array(hidden_states) @ pcamatrix.T + bias
    #print(reduced)
    return(reduced)

def load_pcamatrix(pkl_pca_in):
    with open(pkl_pca_in, "rb") as f:
        cache_pca = pickle.load(f)
        pcamatrix = cache_pca['pcamatrix']
        bias = cache_pca['bias']


    return(pcamatrix, bias)


def save_pcamatrix(pcamatrix, bias, pkl_pca_out):
    with open(pkl_pca_out, "wb") as o:
        pickle.dump({'bias': bias, 'pcamatrix': pcamatrix }, o, protocol=pickle.HIGHEST_PROTOCOL)
   
    pkl_pca_log = "{}.description".format(pkl_pca_out)
    with open(pkl_pca_log, "w") as pout:
        pout.write("Contains objects 'bias' and 'pcamatrix'\n Apply with 'np.array(hidden_states) @ pcamatrix.T + bias'")

def check_target_dimensions(embeddings, target_dim):
        if embeddings.shape[0] < target_dim:
             print("Error: Target dimensions must be smaller than number of embeddings, {} is less than {}".format(embeddings.shape[0], target_dim))
             exit()

def save_embeddings(pkl_out, embedding_name, embeddings_reduced):

   with open(pkl_out, "wb") as o:
       pickle.dump({embedding_name:embeddings_reduced}, o, protocol=pickle.HIGHEST_PROTOCOL)

   pkl_out_log = "{}.description".format(pkl_out)
   with open(pkl_out_log, "w") as pout:
        pout.write("Post PCA object {} dimensions: {}\n".format(embedding_name, embeddings_reduced.shape))


def control_pca(embedding_dict, embedding_name, pkl_pca_in = "", pkl_pca_out = "", target_dim = None, max_train_sample_size = None, pkl_out = ""):


    embeddings = embedding_dict[embedding_name] 
    #PCA takes 2d embeddings.
    if embedding_name == 'aa_embeddings':
          seqlen = embeddings.shape[1]
          embeddings = reshape_3d_to_2d(embeddings)
         

    # If using an already saved PCA matrix + bias, load them in
    if pkl_pca_in:
        pcamatrix,bias = load_pcamatrix(pkl_pca_in)

    # Otherwise train a new one (save if filename provided)
    else:
        check_target_dimensions(embeddings, target_dim)
        pcamatrix, bias = train_pca(embeddings, target = target_dim, max_train_sample_size = max_train_sample_size)

        if pkl_pca_out:
            save_pcamatrix(pcamatrix, bias, pkl_pca_out)

    

    # If outfile for reduced embedding provided, apply pcamatrix + bias and write to file
    if pkl_out:
        embeddings_reduced = apply_pca(embeddings, pcamatrix, bias)

        # For aa_embeddings, need convert back to 3d array
        if embedding_name == 'aa_embeddings':
            embeddings_reduced = reshape_2d_to_3d(embeddings_reduced, seqlen, target_dim)

        save_embeddings(pkl_out, embedding_name, embeddings_reduced)

        return(embeddings_reduced)

if __name__ == "__main__":



    args = get_pca_args()

    with open(args.pkl_in, "rb") as f:
        embedding_dict = pickle.load(f)

    control_pca(embedding_dict,
                args.embedding_name, 
                pkl_pca_in = args.pkl_pca_in, 
                pkl_pca_out = args.pkl_pca_out, 
                target_dim = args.target_dim, 
                max_train_sample_size = args.max_train_sample_size, 
                pkl_out = args.pkl_out)




# Would like to use SentenceTransformers GPU parallelization, but only currently can do sequence embeddings
#def embed_sequences(model_path, sequences, extra_padding,  pkl_out):
#    '''
#    
#    Get a pkl of embeddings for a list of sequences using a particular model
#    Embeddings will have shape xx
#
#    Takes:
#       model_path (str): Path to a particular transformer model
#                         ex. "prot_bert_bfd"
#       sequences (list): List of sequences with a space between each acids.  
#                         ex ["M E T", "S E Q"]
#       pkl_out (str)   : Filename of output pickle of embeddings
# 
#    '''
#    print("Create word embedding model")
#    word_embedding_model = models.Transformer(model_path)
#
#    # Default pooling strategy
#    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
#
#    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
#    print("SentenceTransformer model created")
#    
#    pool = model.start_multi_process_pool()
#
#    # Compute the embeddings using the multi-process pool
#    # about 1.5 hours to this step with 4 GPU and 1.4 million sequences 
#    print("Computing embeddings")
#
#    e = model.encode(sequences, output_value = 'token_embeddings')
#    print(e)
#
#    embeddings = model.encode_multi_process(sequences, pool, output_value = 'token_embeddings')
#
#    print("Embeddings computed. Shape:", embeddings.shape)
#
#    #Optional: Stop the proccesses in the pool
#    model.stop_multi_process_pool(pool)
#
#    return(embeddings)    
   


