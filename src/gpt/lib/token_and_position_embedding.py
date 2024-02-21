from keras.layers import (Embedding,
                          Layer)
from keras import ops


class TokenAndPositionEmbedding(Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        super().__init__()
        self.token_emb = Embedding(input_dim=vocab_size, output_dim=embed_dim)  # shape = vocab_size x embed_dim = 50000*256
        self.pos_emb = Embedding(input_dim=maxlen, output_dim=embed_dim)  # shape = maxlen * embed_dim = 80*256

    def call(self, x):
        maxlen = ops.shape(x)[-1]
        positions = ops.arange(0, maxlen, 1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
