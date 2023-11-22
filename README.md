# Transformer_Encoder
Implementation of Transformer only encoder

# Requirements
torch == 2.0.1

numpy = 1.23.5

transformers = 4.35.2

# Collab installations
!pip install transformers datasets


# Description
The Transformer Encoder Only Model is the first section of the AutoEncoder in the Transformer Model, in this case the transformer encoder model is a feature extractor that does Matrix Multiplication to represent hierarchical understanding of  context in natural language. The architecture starts with an embedding layer to find relations between words, then proceeds on a positional encoder that allows the model to keep words in their particular sequence this means that the architecture takes into consideration the sequence of words(tokens) instead of words(tokens) being disorganized such as bags of words. Then, we have a multihead-attention which over time understand the context of words in paragraph and finally we send it into a feedforward network that will gives us the neccesary outputs of the task we are trying to accomplish and in this case is classifying if a movie review is positive or negative.

# Dataset
Glue Dataset

# Tokenizer
Distilbert

# Architecture
Transfomer Encoder Only Model

# optimizer
Adam

# loss function
Cross Entropy Loss

# Text Result:
![transformer_encoder](https://github.com/Santiagor2230/Transformer_Encoder/assets/52907423/f04cd688-0b5b-4748-857f-21dda883fe90)
