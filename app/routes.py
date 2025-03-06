from flask import Flask, Blueprint, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import pickle
from gensim.models import Word2Vec
from tensorflow.keras.models import Model

app = Flask(__name__)
CORS(app)

main = Blueprint('main', __name__)
CORS(main)

# Define variables globally with None as a fallback
tokenizer_q = None
tokenizer_d = None
lr_model_q = None
lr_model_d = None
feature_extractor_q = None
feature_extractor_d = None
max_length = 100

# Paths
model_path_q = 'app/model/cnn_model.h5'
tokenizer_path_q = 'app/model/tokenizer.pkl'
w2v_model_path_q = 'app/model/w2v_model.bin'
lr_model_path_q = 'app/model/lr_model.pkl'

model_path_d = 'app/model/cnn_model2.h5'
tokenizer_path_d = 'app/model/tokenizer2.pkl'
w2v_model_path_d = 'app/model/w2v_model.bin'
lr_model_path_d = 'app/model/lr_model2.pkl'

try:
    # Load quality models
    combined_model_q = tf.keras.models.load_model(model_path_q)
    w2v_model_q = Word2Vec.load(w2v_model_path_q)
    with open(tokenizer_path_q, 'rb') as f:
        tokenizer_q = pickle.load(f)
    with open(lr_model_path_q, 'rb') as f:
        lr_model_q = pickle.load(f)
    feature_extractor_q = Model(inputs=combined_model_q.input, outputs=combined_model_q.layers[-2].output)

    # Load difficulty models
    combined_model_d = tf.keras.models.load_model(model_path_d)
    w2v_model_d = Word2Vec.load(w2v_model_path_d)
    with open(tokenizer_path_d, 'rb') as f:
        tokenizer_d = pickle.load(f)
    with open(lr_model_path_d, 'rb') as f:
        lr_model_d = pickle.load(f)
    feature_extractor_d = Model(inputs=combined_model_d.input, outputs=combined_model_d.layers[-2].output)

except Exception as e:
    print(f"Error loading models: {e}")

@main.route('/predict', methods=['POST'])
def predict():
    try:
        if tokenizer_q is None or tokenizer_d is None:
            return jsonify({"error": "Models not loaded properly"}), 500

        comments = request.get_json()
        if not comments:
            return jsonify({"error": "Please enter an array."}), 400

        for comment in comments:
            sentence = comment["snippet"]["topLevelComment"]["snippet"]["textOriginal"]

            # Preprocess for quality
            sequence_q = tokenizer_q.texts_to_sequences([sentence])
            padded_sequence_q = tf.keras.preprocessing.sequence.pad_sequences(sequence_q, maxlen=max_length, padding='post')
            extracted_features_q = feature_extractor_q.predict(padded_sequence_q)
            prediction_q = lr_model_q.predict(extracted_features_q)
            quality = int(prediction_q[0] + 1)

            # Preprocess for difficulty
            sequence_d = tokenizer_d.texts_to_sequences([sentence])
            padded_sequence_d = tf.keras.preprocessing.sequence.pad_sequences(sequence_d, maxlen=max_length, padding='post')
            extracted_features_d = feature_extractor_d.predict(padded_sequence_d)
            prediction_d = lr_model_d.predict(extracted_features_d)
            difficulty = int(prediction_d[0] + 1)

            comment["quality"] = quality
            comment["difficulty"] = difficulty
            print(comment)

        return jsonify(comments)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@main.route('/predict_extension', methods=['POST'])
def predict_extension():
    try:
        data = request.get_json()
        comments = data.get("comments", [])

        if not isinstance(comments, list) or len(comments) == 0:
            return jsonify({"error": "Invalid data format"}), 400

        processed_comments = []
        for text in comments[:10]:  # Limit to 10 comments for speed
            sequence = tokenizer_q.texts_to_sequences([text])
            padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=max_length, padding='post')

            extracted_features = feature_extractor_q.predict(padded_sequence)
            prediction = lr_model_q.predict(extracted_features)
            quality = int(prediction[0] + 1)  # Convert NumPy int64 to Python int && Convert 0-indexed to 1-indexed

            processed_comments.append({"text": text, "quality": quality, "difficulty": quality})

        return jsonify(processed_comments)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


app.register_blueprint(main)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
