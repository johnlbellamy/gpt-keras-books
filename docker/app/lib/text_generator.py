from lib.gpt import build_gpt
import json

import numpy as np
from keras import ops
from keras.activations import softmax
from keras.saving import load_model
import yaml


from pathlib import Path



class TextGenerator:
    """A class to generate text from a trained model.
    1. Feed some starting prompt to the model
    2. Predict probabilities for the next token
    3. Sample the next token and add it to the next input1

    Arguments:
        max_tokens: Integer, the number of tokens to be generated after prompt.
        top_k: Integer, sample from the `top_k` token predictions.
        print_every: Integer, print after this many epochs.
    """
    with open(f"config/data.yaml", "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    BATCH_SIZE = config.get("batch_size")
    # Strings shorter than this will be discarded
    MIN_STRING_LEN = config.get("min_string_length")
    SEQ_LEN = config.get("seq_len")
    VOCAB_SIZE = config.get("vocab_size")
    MAX_LEN = config.get("max_len")

    def __init__(
            self,
            model,
            max_tokens: int,
            top_k: int = 10,
    ):
        """
        @type print_every: int
        @type top_k: int
        @type max_tokens: int
        @tpye model: Model
        """

        with open(f"config/vocab.json", "r") as vocab_file:
            self.vocab: [] = json.load(vocab_file)
        self.word_to_index: dict = self.get_word_to_index()
        self.max_tokens: int = max_tokens
        self.start_tokens: list = []
        self.k: int = top_k

        self.model = model

    def sample_from(self, logits):
        """Get sample from model"""
        logits, indices = ops.top_k(logits, k=self.k, sorted=True)
        indices = np.asarray(indices).astype("int32")
        preds = softmax(ops.expand_dims(logits, 0))[0]
        preds = np.asarray(preds).astype("float32")
        return np.random.choice(indices, p=preds)

    def detokenize(self, number):
        """gets word from token number"""
        return self.vocab[number]

    def get_word_to_index(self):
        """ Tokenize starting prompt"""
        word_to_index = {}
        for index, word in enumerate(self.vocab):
            word_to_index[word] = index
        return word_to_index

    def get_start_tokens(self, prompt: str) -> list:
        """Gets tokens for prompt"""
        start_tokens = [self.word_to_index.get(_, 1) for _ in prompt.split()]
        return start_tokens

    def generate(self, prompt_tokens: list, logs=None) -> str:
        """Generates text from book gpt model"""
        prompt_tokens = [_ for _ in prompt_tokens]
        num_tokens_generated = 0
        tokens_generated = []
        while num_tokens_generated <= self.max_tokens:
            pad_len = TextGenerator.MAX_LEN - len(prompt_tokens)
            sample_index = len(prompt_tokens) - 1
            if pad_len < 0:
                x = prompt_tokens[:TextGenerator.MAX_LEN]
                sample_index = TextGenerator.MAX_LEN - 1
            elif pad_len > 0:
                x = prompt_tokens + [0] * pad_len
            else:
                x = prompt_tokens
            x = np.array([x])
            y, _ = self.model.predict(x, verbose=-1)
            sample_token = self.sample_from(y[0][sample_index])
            tokens_generated.append(sample_token)
            prompt_tokens.append(sample_token)
            num_tokens_generated = len(tokens_generated)

        txt = " ".join(
            [self.detokenize(_) for _ in prompt_tokens + tokens_generated]
        )
        return txt


if __name__ == '__main__':
    prompt = "david went to school at"
    # prompt = "the movie beetlejuice is"
    max_tokens = 10
    text_gen = TextGenerator(max_tokens=max_tokens)
    prompt_tokens = text_gen.get_start_tokens(prompt=prompt)
    print(text_gen.generate(prompt_tokens=prompt_tokens))