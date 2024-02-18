import pickle

import numpy as np
import tensorflow as tf
from keras_nlp.samplers import (ContrastiveSampler,
                                TopPSampler)
from keras_nlp.layers import StartEndPacker
from keras.saving import load_model
from keras_nlp.tokenizers import WordPieceTokenizer

SEQ_LEN = 128
MODEL = load_model("../../bin/gpt-movies.keras")
with open("../../bin/vocab.obj", "rb") as vocab_obj:
    VOCAB = pickle.load(vocab_obj)

TOKENIZER = WordPieceTokenizer(
    vocabulary=VOCAB,
    sequence_length=SEQ_LEN,
    lowercase=True,
)
START_PACKER = StartEndPacker(
    sequence_length=SEQ_LEN,
    start_value=TOKENIZER.token_to_id("[BOS]"),
)
BATCH_SIZE = 64
HIDDEN_SIZE = 5
INDEX = 1


def _next(prompt: str, cache, index: int) -> tuple:
    """
    :param index:
    :param cache:
    :type prompt: str
    """
    logits = MODEL(prompt)[:, index - 1, :]
    # Ignore hidden states for now; only needed for contrastive search.
    hidden_states = None
    # prompt_batch_size = tf.shape(prompt)[0]
    # hidden_states = np.ones((prompt_batch_size, HIDDEN_SIZE))
    return logits, hidden_states, cache


if __name__ == '__main__':
    prompt_tokens = START_PACKER(TOKENIZER([""]))
    sampler = TopPSampler(p=0.5)
    # sampler = ContrastiveSampler()

    prompt_batch_size = tf.shape(prompt_tokens)[0]
    hidden_states = np.ones((prompt_batch_size, INDEX, HIDDEN_SIZE))
    output_tokens = sampler(
        next=_next,
        prompt=prompt_tokens,
        index=INDEX)
    txt = TOKENIZER.detokenize(output_tokens)
    print(f"Top-P search generated text: \n{txt}\n")
