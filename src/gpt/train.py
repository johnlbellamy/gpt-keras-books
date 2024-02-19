from gpt import build_gpt
from data.data_books import Embeddings
import json
import tensorflow as tf

mirrored_strategy = tf.distribute.MirroredStrategy()

if __name__ == '__main__':
    embeddings = Embeddings()  # instantiate Embeddings class
    embeddings.prepare_raw_data()  # loads data into tf.data TextLineDataset format
    embeddings.create_vectorize_layer()  # builds the training dataset ready for model
    embeddings.get_training_dataset()

    #with mirrored_strategy.scope():
    model = build_gpt()
    model.summary()
    model_config = model.get_config()
    model_config_json = json.dumps(model_config)
    with open("inference/model_config.json", "w") as model_obj:
        model_obj.write(model_config_json)
    model.fit(embeddings.text_ds,
              epochs=75)

    print("Saving model")
    model.save("gpt-books.keras")
