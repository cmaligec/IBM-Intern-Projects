import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from tensorflow import linalg
from tensorflow.keras import Sequential
from tensorflow.keras.layers import TextVectorization, InputLayer
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D



def run_sk(train_data, vocab_size, trunc_size):

    # Turn the corpus text into a document-term sparse matrix of tfidf values.
    vectorizer = TfidfVectorizer(max_features = vocab_size, lowercase = False)
    vectorized_corpus = vectorizer.fit_transform(train_data)

    # Obtain the corpus vocabulary.
    vocab = vectorizer.get_feature_names_out()
    len_vocab = len(vocab)
    
    # Apply Truncated Singular Value Decomposition (Truncated SVD) to approximate the document-term sparse matrix in order to reduce its dimensionality and speed up processing time. The tf-idf document-term matrix (M) is approximately decomposed into U*Sigma*V_transpose.
    # If M is an mxn matrix, with m = num_documents and n = num_terms, then the three components have the following dimensions: U is mxk, Sigma is kxk, and V_transpose is kxn. The latent dimension, k = the number of features after truncation (trunc_size).

     
    # Reduce the dimensionality of the document-term sparse matrix.
    lsa_obj = TruncatedSVD(n_components=trunc_size, random_state=123)
    lsa_obj.fit_transform(vectorized_corpus)

    # Obtain feature-term matrix of shape trunc_size x len_vocab. This can then be used to define word embeddings with a much lower number of dimensions.
    v = lsa_obj.components_

    #Obtain term-feature matrix of shape len_vocab x trunc_size.
    v = np.transpose(v)
    
    return vocab, v


def run_st(text_data, val_data, logs_path, vocab_size, trunc_size, use_profiler=False):
    
    # Use the text vectorization layer to map strings to tf-idf values (float). 
    vectorize_layer = TextVectorization(
        standardize= None,
        split = None,
        max_tokens=vocab_size,
        output_mode='tf_idf'
        )

    # Call adapt to build the vocabulary.
    vectorize_layer.adapt(text_data)
    vocab = vectorize_layer.get_vocabulary()
    len_vocab = len(vocab)
    
    # Create a model, not for training or testing, but to apply TextVectorization to the text data.
    model = Sequential([
        InputLayer(input_shape=(1,), dtype=tf.string),
        vectorize_layer
        ])
    print('model created')   
    
    # Compile model.
    model.compile()
    print('model compiled')
    
    # Train model.
    if(use_profiler):
    
        # To make use of the TensorBoard Profiler to analyze results for further optimization. Callbacks can only be included in model fits, which is another reason for having created a model to vectorize the text data.
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=logs_path, 
            profile_batch=(1,50)
            )
        
        model.fit(
            text_data,
            validation_data=val_data,
            epochs=1,
            callbacks=[tensorboard_callback]
            )
    else:
        model.fit(
            text_data,
            epochs=1
            )
    print('model fitted')
    # model.predict() outputs the vectorized version of the text data. This was the main reason for using a model.
    vectorized_corpus = model.predict(text_data)
    print('vectorized corpus created')

    # Apply Thin Singular Value Decomposition (Thin SVD) to approximate the document-term sparse matrix in order to reduce its dimensionality and speed up processing time. The tf-idf document-term matrix (M) is decomposed into U*Sigma*V_transpose.
    # If M is an mxn matrix, with m = num_documents and n = num_terms, then the three components have the following dimensions: U is mxk, Sigma is kxk, and v_transpose is kxn where k = min{m, n}. We can call k the number of features.
    # The term-feature matrix, which is just v with dimensions nxk, can then be used to define word embeddings with a much lower number of dimensions.
    s, u, v = tf.linalg.svd(vectorized_corpus)
    print('SVD performed')
    # Truncate v to lower dimensions, so that v has dimensions nxt with t << k.
    num_desired_features = 20
    v = v[:, 0:num_desired_features]
    print('v truncated')
    print('results')
    return vocab, v


def run_dp(train_data, text_data, val_data, logs_path, vocab_size, trunc_size, use_profiler=False):
    sequence_length = 100
    
    # Use the text vectorization layer to map strings to integers.
    # Set maximum_sequence length as all samples are not of the same length.
    vectorize_layer = TextVectorization(
        standardize= None,
        max_tokens=vocab_size,
        output_mode='int',
        output_sequence_length=sequence_length
        )


    # Call adapt to build the vocabulary.
    vectorize_layer.adapt(text_data)
    vocab = vectorize_layer.get_vocabulary()
    len_vocab = len(vocab)
    
    # Creating model.
    embedding_dim=20
    model = Sequential([
      vectorize_layer,
      Embedding(vocab_size, embedding_dim, name="embedding"),
      GlobalAveragePooling1D(),
      Dense(embedding_dim, activation='relu'),
      Dense(1)
    ])  

    # Compile model.
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Train model.
    if(use_profiler):
    
        # To make use of the TensorBoard Profiler to analyze results for further optimization. Callbacks can only be included in model fits, which is another reason for having created a model to vectorize the text data.
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=logs_path, 
            profile_batch=(1,50)
            )
        
        model.fit(
            train_data,
            validation_data=val_data,
            epochs=10,
            callbacks=[tensorboard_callback]
            )
    else:
        model.fit(
            train_data,
            epochs=10
            )

    # Get the embeddings.
    v = model.get_layer('embedding').get_weights()[0]

    return vocab, v
