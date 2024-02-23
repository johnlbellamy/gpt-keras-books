from keras.layers import (Input,
                          Dense)
import yaml
from keras import Model
from keras.losses import SparseCategoricalCrossentropy

from lib.token_and_position_embedding import TokenAndPositionEmbedding
from lib.transformer_block import TransformerBlock

from pathlib import Path

config_path = str(Path(__file__).parents[1])

with open(f"/serving/app/config/config.yaml", "r") as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

VOCAB_SIZE = config.get("vocab_size")
EMBED_DIM = config.get("embed_dim")
NUM_HEADS = config.get("num_heads")
FEED_FORWARD_DIM = config.get("feed_forward_dim")
MAX_LEN = config.get("max_len")


def build_gpt() -> Model:
    """Returns a keras model"""
    inputs = Input(shape=(MAX_LEN,), dtype="int32")
    embedding_layer = TokenAndPositionEmbedding(MAX_LEN, VOCAB_SIZE, EMBED_DIM)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(EMBED_DIM, NUM_HEADS, FEED_FORWARD_DIM)
    x = transformer_block(x)
    outputs = Dense(VOCAB_SIZE)(x)
    model = Model(inputs=inputs, outputs=[outputs, x])
    loss_fn = SparseCategoricalCrossentropy(from_logits=True)
    model.compile(
        "adam",
        loss=[loss_fn, None],
    )  # No loss and optimization based on word embeddings from transformer block
    return model
