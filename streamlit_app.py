import streamlit as st
import pandas as pd
import numpy as np
import fasttext
import pytesseract
from pinecone import Pinecone
from PIL import Image
import tensorflow as tf
import os
import pickle

# Initialize Pinecone
pc = Pinecone(api_key="API-key")
index_name = 'products'
index = pc.Index(index_name)

# Load FastText model
fasttext_model = fasttext.load_model('./fasttext_model.bin')

# Load CNN model and label encoder
cnn_model = tf.keras.models.load_model('./cnn_model.h5')
with open('./label_encoder.pkl', 'rb') as f:
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
    # Predict the image class
    image = preprocess_image(image_path)
    predictions = cnn_model.predict(image)
    predicted_class = np.argmax(predictions, axis=1)
    predicted_label = label_encoder.inverse_transform(predicted_class)[0]

    # Query Pinecone using the predicted label (Stock ID)
    response = pinecone_index.query(
        namespace='',  # Specify namespace if used
        id=str(predicted_label),  # Use the predicted label as the query id
        top_k=10,
        include_values=True,
        include_metadata=True
    )

    return response

# Streamlit app
st.title('E-coms Product Explorer')

# Page Navigation
st.sidebar.title('Navigation')
page = st.sidebar.radio('How would you like to get product recommendations?', ['Text Query Interface', 'Image Query Interface', 'Product Image Upload Interface'])

if page == 'Text Query Interface':
    st.header('Text Query Interface')
    query = st.text_input('Enter your query:').upper()
    if query:
        query_vector = embed_query(query, fasttext_model)
        pinecone_results = index.query(
            vector=query_vector.tolist(),  # Convert the query vector to a list
            top_k=10,  # Number of top results to return
            include_values=True,  # Include the values in the response
            include_metadata=True  # Include metadata in the response
        )
        
        if pinecone_results['matches']:
            st.write("Top Products:")
            products = []
            for match in pinecone_results['matches']:
                product_id = match['id']
                
                if 'metadata' in match:
                    description = match['metadata'].get('description', 'N/A')
                    #score = match['score']
                else:
                    description = 'N/A'
                product = {
                    'Product ID': product_id,
                    #'Score': score,
                    'Description': description
                }
                products.append(product)
            products_df = pd.DataFrame(products)
            st.table(products_df)
        else:
            st.write("No products found.")

elif page == 'Image Query Interface':
    st.header('Image Query Interface')
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Extracting text...")

        # Extract text from uploaded image
        extracted_text = extract_text_from_image(uploaded_file).upper()
        st.write(f"Extracted Text: {extracted_text}")
        
        # Embed the extracted text query
        if extracted_text:
            query_vector = embed_query(extracted_text, fasttext_model)
            pinecone_results = index.query(
                vector=query_vector.tolist(),  # Convert the query vector to a list
                top_k=10,  # Number of top results to return
                include_values=True,  # Include the values in the response
                include_metadata=True  # Include metadata in the response
            )
            
            if pinecone_results['matches']:
                st.write("Top Products:")
                products = []
                for match in pinecone_results['matches']:
                    product_id = match['id']
                    #score = match['score']
                    if 'metadata' in match:
                        description = match['metadata'].get('description', 'N/A')
                    else:
                        description = 'N/A'
                    product = {
                        'Product ID': product_id,
                        #'Score': score,
                        'Description': description
                    }
                    products.append(product)
                products_df = pd.DataFrame(products)
                st.table(products_df)
            else:
                st.write("No products found.")
        else:
            st.write("No text extracted from the image.")

elif page == 'Product Image Upload Interface':
    st.header('Product Image Upload Interface')
    uploaded_image = st.file_uploader("Upload a product image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Product Image.', use_column_width=True)
        st.write("")
        st.write("Identifying...")

        # Predict using the CNN model
        try:
            # Preprocess the image for prediction
            image_array = tf.keras.preprocessing.image.img_to_array(image)
            image_array = tf.image.resize(image_array, (128, 128)) / 255.0
            image_array = np.expand_dims(image_array, axis=0)

            # Predict using the CNN model
            predictions = cnn_model.predict(image_array)
            predicted_index = np.argmax(predictions, axis=1)[0]
            predicted_label = label_encoder.inverse_transform([predicted_index])[0]

            # Display predicted label
            st.write(f"Identified Product ID: {predicted_label}")

            # Query Pinecone using the predicted label (Stock ID)
            response = index.query(
                namespace='',  # Specify namespace if used
                id=str(predicted_label),  # Use the predicted label as the query id
                top_k=10,
                include_values=True,
                include_metadata=True
            )

            # Display the top related products
            if response['matches']:
                st.write("Top Related Products:")
                products = []
                for match in response['matches']:
                    product_id = match['id']
                    #score = match['score']
                    if 'metadata' in match:
                        description = match['metadata'].get('description', 'N/A')
                    else:
                        description = 'N/A'
                    product = {
                        'Product ID': product_id,
                        #'Score': score,
                        'Description': description
                    }
                    products.append(product)
                products_df = pd.DataFrame(products)
                st.table(products_df)
            else:
                st.write("No related products found.")
        except Exception as e:
            st.write(f"Error: {str(e)}")
