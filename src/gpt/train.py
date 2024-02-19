from gpt import build_gpt
from data.data_books import Embeddings
import json
import tensorflow as tf
import pickle
from pathlib import Path


config_path = str(Path(__file__).parents[1])
mirrored_strategy = tf.distribute.MirroredStrategy()

if __name__ == '__main__':
    embeddings = Embeddings()  # instantiate Embeddings class
    embeddings.prepare_raw_data()  # loads data into tf.data TextLineDataset format
    # builds the training dataset ready for model
    embeddings.create_vectorize_layer()
    embeddings.get_training_dataset()

    json_obj = json.dumps(embeddings.vocab)
    with open("config/vocab.json", "w") as file:
        file.write(json_obj)

    # with mirrored_strategy.scope():
    model = build_gpt()
    model.summary()
    model.fit(embeddings.text_ds,
              epochs=25,
              steps_per_epoch=len(list(embeddings.text_ds.as_numpy_iterator()))//128)

    print("Saving model")
    model.save_weights("bin/gpt-books.weights.h5", overwrite=True)
