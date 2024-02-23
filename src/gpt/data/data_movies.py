import os
import string
from keras.layers import TextVectorization
from tensorflow.data import TextLineDataset
import tensorflow.strings as tf_strings
import tensorflow as tf
import yaml
import random
tf.random.set_seed(56)
mirrored_strategy = tf.distribute.MirroredStrategy()

CONFIG_PATH = "../../"


class Encodings:
    with open(f"config/config.yaml", "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # Strings shorter than this will be discarded
    MIN_STRING_LEN = config.get("min_string_length")
    SEQ_LEN = config.get("seq_len")
    VOCAB_SIZE = config.get("vocab_size")
    MAX_LEN = config.get("max_len")
    BATCH_SIZE = config.get("batch_size")

    def __init__(self):
        self.text_ds: TextLineDataset = None
        self.vectorize_layer: TextVectorization = None
        self.vocab: list = []

    def prepare_raw_data(self):
        # The dataset contains each review in a separate text file
        # The text files are present in four different folders
        # Create a list all files

        print(os.getcwd())
        filenames = []
        directories = [
            f"{CONFIG_PATH}data/aclImdb/train/pos",
            f"{CONFIG_PATH}data/aclImdb/train/neg",
            f"{CONFIG_PATH}data/aclImdb/test/pos",
            f"{CONFIG_PATH}data/aclImdb/test/neg",
        ]
        for dir in directories:
            for f in os.listdir(dir):
                filenames.append(os.path.join(dir, f))

        print(f"{len(filenames)} files")
        # Create a dataset from text files
        random.shuffle(filenames)
        text_ds = TextLineDataset(filenames)
        text_ds = text_ds.shuffle(buffer_size=256)
        self.text_ds = text_ds.batch(Encodings.BATCH_SIZE)


    def create_vectorize_layer(self):
        """Create a vectorization layer and adapt it to the text"""
        self.vectorize_layer = TextVectorization(
            standardize="lower_and_strip_punctuation",
            max_tokens=Encodings.VOCAB_SIZE - 1,
            output_mode="int",
            output_sequence_length=Encodings.MAX_LEN + 1,
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
        with mirrored_strategy.scope():
            self.text_ds = self.text_ds.map(self.prepare_lm_inputs_labels,
                                            num_parallel_calls=tf.data.AUTOTUNE)
            self.text_ds = self.text_ds.prefetch(tf.data.AUTOTUNE)
