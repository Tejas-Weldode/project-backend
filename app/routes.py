from flask import Flask, Blueprint, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import pickle
from gensim.models import Word2Vec
from tensorflow.keras.models import Model

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

main = Blueprint('main', __name__)
CORS(main)  # Ensure CORS is applied to the blueprint

# Load models and tokenizer
model_path_q = 'app/model/cnn_model.h5'
tokenizer_path_q = 'app/model/tokenizer.pkl'
w2v_model_path_q = 'app/model/w2v_model.bin'
lr_model_path_q = 'app/model/lr_model.pkl'
model_path_d = 'app/model/cnn_model2.h5'
tokenizer_path_d = 'app/model/tokenizer2.pkl'
w2v_model_path_d = 'app/model/w2v_model2.bin'
lr_model_path_d = ' app/model/lr_model2.pkl'
try:
    combined_model_q = tf.keras.models.load_model(model_path_q)
    w2v_model_q = Word2Vec.load(w2v_model_path_q)

    combined_model_d = tf.keras.models.load_model(model_path_d)
    w2v_model_d = Word2Vec.load(w2v_model_path_d)

    with open(tokenizer_path_q, 'rb') as f:
        tokenizer_q = pickle.load(f)
    with open(tokenizer_path_d, 'rb') as f:
        tokenizer_d = pickle.load(f)

    with open(lr_model_path_q, 'rb') as f:
        lr_model_q = pickle.load(f)
    with open(lr_model_path_d, 'rb') as f:
        lr_model_d = pickle.load(f)

    feature_extractor_q = Model(inputs=combined_model_q.input, outputs=combined_model_q.layers[-2].output)
    feature_extractor_d = Model(inputs=combined_model_d.input, outputs=combined_model_d.layers[-2].output)

    max_length = 100
except Exception as e:
    print(f"Error loading models: {e}")

@main.route('/predict', methods=['POST'])
def predict():
    try:
        comments = request.get_json()

        if not comments:
            return jsonify({"error": "Please enter an array."}), 400

        for comment in comments:

            sentence = comment["snippet"]["topLevelComment"]["snippet"]["textOriginal"]

            sequence_q = tokenizer_q.texts_to_sequences([sentence])
            padded_sequence_q = tf.keras.preprocessing.sequence.pad_sequences(sequence_q, maxlen=max_length, padding='post')

            sequence_d = tokenizer_d.texts_to_sequences([sentence])
            padded_sequence_d = tf.keras.preprocessing.sequence.pad_sequences(sequence_d, maxlen=max_length, padding='post')

            extracted_features_q = feature_extractor_q.predict(padded_sequence_q)
            prediction_q = lr_model_q.predict(extracted_features_q)
            quality = int(prediction_q[0] + 1)  # Convert NumPy int64 to Python int && Convert 0-indexed to 1-indexed

            extracted_features_d = feature_extractor_d.predict(padded_sequence_d)
            prediction_d = lr_model_d.predict(extracted_features_d)
            difficulty = int(prediction_d[0] + 1)  # Convert NumPy int64 to Python int && Convert 0-indexed to 1-indexed

            # Add quality field to the comment object
            comment["quality"] = quality
            comment["difficulty"] = difficulty
            print(comment)

        return jsonify(comments)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

app.register_blueprint(main)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)