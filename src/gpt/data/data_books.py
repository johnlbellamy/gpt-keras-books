from pathlib import Path
import os
import string
from tensorflow.data import TextLineDataset
from keras.layers import TextVectorization
import tensorflow.strings as tf_strings
import tensorflow as tf
import yaml

tf.random.set_seed(56)
mirrored_strategy = tf.distribute.MirroredStrategy()
config_path = str(Path(__file__).parents[1])
data_path = str(Path(__file__).parents[3])


class Embeddings:
    print(f"{config_path}")
    with open(f"{config_path}/config/config.yaml", "r") as stream:
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

    def __init__(self):
        self.text_ds: TextLineDataset = None
        self.vectorize_layer: TextVectorization = None
        self.vocab: list = []

    def prepare_raw_data(self):
        """Builds the training and validation datasets"""
        print("Loading training data...")
        self.text_ds = (
            TextLineDataset(f"{data_path}/data/simplebooks/train_small.txt")
            .batch(Embeddings.BATCH_SIZE)
            .shuffle(buffer_size=256)
        )

    @staticmethod
    def custom_standardization(input_string):
        """Remove html line-break tags and handle punctuation"""
        lower_cased = tf_strings.lower(input_string)
        return tf_strings.regex_replace(lower_cased, f"([{string.punctuation}])", r" \1")

    def create_vectorize_layer(self):
        """Create a vectorization layer and adapt it to the text"""
        self.vectorize_layer = TextVectorization(
            standardize="lower_and_strip_punctuation",
            max_tokens=Embeddings.VOCAB_SIZE - 1,
            output_mode="int",
            output_sequence_length=Embeddings.MAX_LEN + 1,
        )
        self.vectorize_layer.adapt(self.text_ds)
        # To get words back from token indices
        self.vocab = self.vectorize_layer.get_vocabulary()

    def prepare_lm_inputs_labels(self, text: str) -> tuple:
        """
        Shift word sequences by 1 position so that the target for position (i) is
        word at position (i+1). The model will use all words up till position (i)
        to predict the next word.
        """
        text = tf.expand_dims(text, -1)
        tokenized_sentences = self.vectorize_layer(text)
        x = tokenized_sentences[:, :-1]
        y = tokenized_sentences[:, 1:]
        return x, y

    def get_training_dataset(self):
        # with mirrored_strategy.scope():
        self.text_ds = self.text_ds.map(self.prepare_lm_inputs_labels,
                                        num_parallel_calls=tf.data.AUTOTUNE)
        self.text_ds = self.text_ds.prefetch(tf.data.AUTOTUNE)
