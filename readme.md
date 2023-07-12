# Word2Vec Implementation

This project implements the Word2Vec model from scratch, including both the Skip-gram and Continuous Bag of Words (CBOW) variants. Word2Vec is a popular technique in natural language processing for learning word embeddings, which represent words as dense vectors in a continuous vector space.

## Contents

- [Overview](#overview)
- [Implementation Details](#implementation-details)
- [Usage](#usage)
- [Demonstration](#demonstration)
- [Dependencies](#dependencies)
- [References](#references)

## Overview

The Word2Vec model is trained on a large corpus of text data to learn the semantic relationships between words. It learns to map words to vectors, where similar words are represented by vectors that are close together in the vector space.

This implementation supports both the Skip-gram and CBOW architectures. The Skip-gram model predicts the context words given a target word, while CBOW predicts the target word based on the context words. The training process involves updating the word vectors using stochastic gradient descent and negative sampling.

## Implementation Details

The implementation consists of the following files:

- `word2vec.py`: Contains the implementation of the Word2Vec model, including both Skip-gram and CBOW architectures.


## Demonstration

Below is an image demonstrating the Word2Vec model's architecture and how it learns word embeddings:

![Word2Vec Demonstration](https://miro.medium.com/v2/resize:fit:1400/format:webp/0*3DFDpaXoglalyB4c.png)
