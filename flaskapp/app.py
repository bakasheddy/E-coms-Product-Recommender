from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import fasttext
import pytesseract
from pinecone import Pinecone
from PIL import Image
import tensorflow as tf
import os
import pickle

# Initialize Flask app
app = Flask(__name__)

# Initialize Pinecone
pc = Pinecone(api_key="1b558c37-2524-47f9-9576-0a2efc720fe8")
index_name = 'products'
pc_index = pc.Index(index_name)

# Load FastText model
fasttext_model = fasttext.load_model('../fasttext_model.bin')

# Load CNN model and label encoder
cnn_model = tf.keras.models.load_model('../cnn_model.h5')
with open('../label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Function to extract text from image using Tesseract OCR
def extract_text_from_image(image):
    img = Image.open(image)
    text = pytesseract.image_to_string(img)
    return text

# Function to embed query using FastText model
def embed_query(query, model):
    tokenized_query = query.split()
    query_vector = sum(model.get_word_vector(word) for word in tokenized_query) / len(tokenized_query)
    return query_vector

# Function to preprocess the image
def preprocess_image(image_path, image_size=(128, 128)):
    image = Image.open(image_path).convert('RGB')
    image = image.resize(image_size)
    image = np.array(image) / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to predict image class and query Pinecone
def process_and_query(image_path, cnn_model, label_encoder, pinecone_index):
    image = preprocess_image(image_path)
    predictions = cnn_model.predict(image)
    predicted_class = np.argmax(predictions, axis=1)
    predicted_label = label_encoder.inverse_transform(predicted_class)[0]

    response = pinecone_index.query(
        namespace='',
        id=str(predicted_label),
        top_k=10,
        include_values=True,
        include_metadata=True
    )
    return response

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/text_query', methods=['GET', 'POST'])
def text_query():
    if request.method == 'POST':
        query = request.form['query'].upper()
        query_vector = embed_query(query, fasttext_model)
        pinecone_results = pc_index.query(
            vector=query_vector.tolist(),
            top_k=10,
            include_values=True,
            include_metadata=True
        )
        products = []
        if pinecone_results['matches']:
            for match in pinecone_results['matches']:
                product_id = match['id']
                description = match['metadata'].get('description', 'N/A') if 'metadata' in match else 'N/A'
                products.append({'Product ID': product_id, 'Description': description})
        return render_template('text_query.html', products=products)
    return render_template('text_query.html', products=None)

@app.route('/image_query', methods=['GET', 'POST'])
def image_query():
    if request.method == 'POST':
        try:
            file = request.files['file']
            if file:
                # Extract text from uploaded image
                extracted_text = extract_text_from_image(file).upper()
                # Embed the extracted text query
                query_vector = embed_query(extracted_text, fasttext_model)
                # Query Pinecone with the embedded query
                pinecone_results = pc_index.query(
                    vector=query_vector.tolist(),
                    top_k=10,
                    include_values=True,
                    include_metadata=True
                )
                products = []
                if pinecone_results['matches']:
                    for match in pinecone_results['matches']:
                        product_id = match['id']
                        description = match['metadata'].get('description', 'N/A') if 'metadata' in match else 'N/A'
                        products.append({'Product ID': product_id, 'Description': description})
                return render_template('image_query.html', products=products, extracted_text=extracted_text)
        except Exception as e:
            print(f"Error processing file: {e}")
            return "Error processing file", 400
    return render_template('image_query.html', products=None)


@app.route('/upload_image', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            image_path = os.path.join('./static/uploads', file.filename)
            file.save(image_path)
            response = process_and_query(image_path, cnn_model, label_encoder, pc_index)
            products = []
            if response['matches']:
                for match in response['matches']:
                    product_id = match['id']
                    description = match['metadata'].get('description', 'N/A') if 'metadata' in match else 'N/A'
                    products.append({'Product ID': product_id, 'Description': description})
            return render_template('upload_image.html', products=products, image_path=image_path)
    return render_template('upload_image.html', products=None)

if __name__ == '__main__':
    app.run(debug=True)
