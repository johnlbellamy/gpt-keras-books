from gpt import build_gpt
from data.data_books import Encodings
import json
import tensorflow as tf
import pickle
from pathlib import Path


config_path = str(Path(__file__).parents[1])
mirrored_strategy = tf.distribute.MirroredStrategy()

if __name__ == '__main__':
    encodings = Encodings()  # instantiate Encodings class
    encodings.prepare_raw_data()  # loads data into tf.data TextLineDataset format
    # builds the training dataset ready for model
    encodings.create_vectorize_layer()
    encodings.get_training_dataset()

    json_obj = json.dumps(encodings.vocab)
    with open("config/vocab.json", "w") as file:
        file.write(json_obj)

    # with mirrored_strategy.scope():
    model = build_gpt()
    model.summary()
    model.fit(encodings.text_ds,
              epochs=100,
              steps_per_epoch=len(list(encodings.text_ds.as_numpy_iterator()))//128)
    print("Saving model")
    model_json = model.to_json()
    with open("bin/model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("bin/gpt-books.weights.h5", overwrite=True)
