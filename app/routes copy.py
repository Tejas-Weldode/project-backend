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
model_path = 'app/model/cnn_model.h5'
tokenizer_path = 'app/model/tokenizer.pkl'
w2v_model_path = 'app/model/w2v_model.bin'
lr_model_path = 'app/model/lr_model.pkl'

try:
    combined_model = tf.keras.models.load_model(model_path)
    w2v_model = Word2Vec.load(w2v_model_path)

    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)

    with open(lr_model_path, 'rb') as f:
        lr_model = pickle.load(f)

    feature_extractor = Model(inputs=combined_model.input, outputs=combined_model.layers[-2].output)
    max_length = 100
except Exception as e:
    print(f"Error loading models: {e}")

@main.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        sentence = data.get('sentence', '')

        if not sentence:
            return jsonify({"error": "Please enter a sentence."}), 400

        sequence = tokenizer.texts_to_sequences([sentence])
        padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=max_length, padding='post')

        extracted_features = feature_extractor.predict(padded_sequence)
        prediction = lr_model.predict(extracted_features)
        quality = prediction[0] + 1  # Convert 0-indexed to 1-indexed

        return jsonify({"sentence": sentence, "quality": int(quality)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

app.register_blueprint(main)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
