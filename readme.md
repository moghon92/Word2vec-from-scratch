# Word2Vec Implementation

This project implements the Word2Vec model from scratch, including both the Skip-gram and Continuous Bag of Words (CBOW) variants. Word2Vec is a popular technique in natural language processing for learning word embeddings, which represent words as dense vectors in a continuous vector space.

## Contents

- [Overview](#overview)
- [Implementation Details](#implementation-details)
- [Demonstration](#demonstration)
  
## Overview

The Word2Vec model is trained on a large corpus of text data to learn the semantic relationships between words. It learns to map words to vectors, where similar words are represented by vectors that are close together in the vector space. Word2vec is a method to efficiently create word embeddings. More details on word2vec and the intuition behind it can be found here :  
* [The Illustrated Word2vec by Jay Alammar](https://jalammar.github.io/illustrated-word2vec/)

This implementation supports both the Skip-gram and CBOW architectures:

* CBOW (Continuous Bag-of-Words) — a model that predicts a current word based on its context words.
* Skip-Gram — a model that predicts context words based on the current word.

## Implementation Details

The implementation consists of the following files:

- `word2vec.py`: Contains the implementation of the Word2Vec model, including both Skip-gram and CBOW architectures.



## Demonstration

### A high level overview of the CBOW model can be described as :    
<p align="center"><img src="https://miro.medium.com/max/1400/1*ETcgajy5s0KNIfMgE5xOqg.png" width="75%" align="center"></p>

CBOW model takes several words, each goes through the same Embedding layer, and then word embedding vectors are averaged before going into the Linear layer.

We will be implementing this model using the architecture described below :    

<p align="center"><img src="https://miro.medium.com/max/1400/1*mLDM3PH12CjhaFoUm5QTow.png" width="75%" align="center"></p>

Here are the steps that needs to be followed for implementing CBOW model :    
* Step-1: Create vocabulary
  * Split each words into tokens.
  * Assign a unique ID to each unique token.

* Step-2: Create CBOW Embeddings
  * Create CBOW embeddings by taking context as N past words and N future words.

* Step-3: Implement CBOW Model
  * Implement CBOW model as described in the architecture above.
 
### A high level overview of the SkipGram model can be described as :    
<p align="center"><img src="https://miro.medium.com/max/720/1*SVs6xTpD7AYviP24UTOYUA.png" width="75%" align="center"></p>

The Skip-Gram model takes a single word as compared to CBOW model.

We will be implementing this model using the architecture described below :    

<p align="center"><img src="https://miro.medium.com/max/720/1*eHh1_t8Wms_hqDNBLuAnFg.png" width="75%" align="center"></p>

Here are the steps that needs to be followed for implementing SkipGram model :    
* Step-1: Create vocabulary
  * Split each words into tokens.
  * Assign a unique ID to each unique token.

* Step-2: Create SkipGram Embeddings
  * Create SkipGram embeddings by taking context as middle word.

* Step-3: Implement SkipGram Model
  * Implement SkipGram model as described in the architecture above. Output SkipGram embeddings for N past words and N future words.
  
