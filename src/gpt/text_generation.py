import string

from gpt import build_gpt
import json
import numpy as np
import keras_nlp

import yaml
from keras_nlp.src.backend import ops
from keras_nlp.src.backend import random
from pathlib import Path

config_path = str(Path(__file__).parents[0])


class TextGenerator:
    """A class to generate text from a trained model.
    1. Feed some starting prompt to the model
    2. Predict probabilities for the next token
    3. Sample the next token and add it to the next input
    @param k: Integer, sample from the `top_k` token predictions.
    """
    with open(f"{config_path}/config/config.yaml", "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    BATCH_SIZE = config.get("batch_size")
    SEQ_LEN = config.get("seq_len")
    VOCAB_SIZE = config.get("vocab_size")
    MAX_LEN = config.get("max_len")

    def __init__(
            self,
            k: int = 10,
            p: float = 50.0
    ):
        """
        @type k: int
        """

        self.p = p
        self.seed_generator = None
        with open(f"{config_path}/config/vocab.json", "r") as vocab_file:
            self.vocab: [] = json.load(vocab_file)
        self.word_to_index: dict = self.get_word_to_index()
        self.start_tokens: list = []
        self.k = k

        self.model = build_gpt()
        self.model.load_weights(
            filepath=f"{config_path}/bin/gpt-movies.weights.h5")
        self.model.summary()
        self.sampler = keras_nlp.samplers.TopPSampler()
        self.seen = []

    @staticmethod
    def remove_punctuation(input_string):
        translator = str.maketrans("", "", string.punctuation)
        clean_string = input_string.translate(translator)
        return clean_string

    def get_next_token(self, probabilities, sample_index):
        """Return a Top P sample of logits"""
        sorted_preds, sorted_indices = ops.top_k(
            probabilities[0][sample_index],
            k=self.k,
            sorted=True
        )
        # Calculate cumulative probability distribution.
        cumulative_probabilities = ops.cumsum(sorted_preds, axis=-1)

        # Create a mask for the tokens to keep.
        keep_mask = cumulative_probabilities <= self.p
        keep_mask = np.array([keep_mask])
        # Shift to include the last token that exceed p.
        shifted_keep_mask = ops.concatenate(
            [ops.ones_like(keep_mask[:, :1]), keep_mask[:, :-1]], axis=-1
        )
        # Filter out unmasked tokens and sample from filtered distribution.
        output_probabilities = ops.where(
            shifted_keep_mask,
            sorted_preds,
            ops.zeros(ops.shape(sorted_preds), dtype=sorted_preds.dtype),
        )
        sorted_next_token = random.categorical(
            ops.log(output_probabilities),
            1,
            seed=self.seed_generator,
            dtype="int32",
        )
        output = ops.take_along_axis(np.array([sorted_indices]), sorted_next_token, axis=-1)
        squeezed = int(output.numpy()[0][0])
        if squeezed and squeezed not in self.seen:
            self.seen.append(squeezed)
            return squeezed
        else:
            # If word is in seen, then run again
            return self.get_next_token(probabilities=probabilities,
                                       sample_index=sample_index)

    def detokenize(self, number):
        """gets word from token number"""
        if isinstance(number, int):
            try:
                res = self.vocab[number]
                if res:
                    return res
                else:
                    raise IndexError
            except IndexError:
                print("Key not found")

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
        num_tokens_generated = 0
        tokens_generated = []
        while num_tokens_generated <= self.k:
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
            probabilities, _ = self.model.predict(x, verbose=-1)
            sample_token = self.get_next_token(probabilities, sample_index)

            tokens_generated.append(sample_token)
            prompt_tokens.append(sample_token)
            num_tokens_generated = len(tokens_generated)

        final_prompt_tokens = [x for x in prompt_tokens if x]
        prompt_txt_list = [self.detokenize(_) for _ in final_prompt_tokens if self.detokenize(_) != "[UNK]"]
        prompt_txt_list = [x for x in prompt_txt_list if x and x != " " and x != ""]
        generated_text = " ".join(prompt_txt_list)
        generated_text = generated_text.replace("  ", " ")
        generated_text = generated_text.replace("   ", " ")
        return generated_text


if __name__ == '__main__':
    prompt = "this movie is"
    top_k = 10
    text_gen = TextGenerator(k=top_k)
    prompt_tokens = text_gen.get_start_tokens(prompt=prompt)
    txt = text_gen.generate(prompt_tokens)
    print(f"Top p search generated text: \n{txt}\n")
