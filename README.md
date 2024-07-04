# E-coms Product Recommender

## About
This project involves the development of a comprehensive recommendation system for an e-commerce platform, integrating natural language processing, vector similarity search, and computer vision. FastText was utilized to embed product descriptions into vector representations, which were then indexed in Pinecone, a scalable and efficient vector database. Additionally, a CNN model was implemented to detect and classify products based on images, enhancing the recommendation capabilities. The system allows for querying these vectors with natural language inputs and image inputs, efficiently returning the top 3 similar products. This dual approach significantly improves the user experience by providing highly relevant product suggestions based on both text descriptions and visual content. The project demonstrates expertise in machine learning, NLP, computer vision, and scalable vector search technologies, showcasing the ability to create sophisticated and effective recommendation systems.

link to the scraped image data is [here](https://drive.google.com/drive/folders/1F4YVmGjnaF63L6JrsYvwAXVaeHd7ej5L?usp=drive_link)

![image](ecoms.gif)
### module 1
- The data was loaded using pandas and a customized output dataframe was provide after each feature was thoroughly examined for anomalys
- Regular expression was utilzedd for the cleaning of features which had repeated patterns of untidiness
- Features was converted to there appropriate data types
- Duplicates and NAN values that could not be helped was dropped and a clean_data was downloaded
  
- Pinecone was installed and access was granted using my api key
- Given the nature of product descriptions in the dataset, which often contain unique terms, brand names, and variations in spelling, FastText is generally the better choice for this task. Its ability to handle out-of-vocabulary words and generate embeddings based on subword information provides a significant advantage.
- Using the fasttext model a vector representations for each product description was created.
- The 'StockCode' feature was used as the products ids and a pincone index was created with the name 'products' and cosine similarity was used for the matric because It efficiently captures the similarity between vectors while mitigating the effects of varying vector magnitudes.
- The index was connected and a function was made to preprocess a user's query
- The function was made to recommend products similar to the user's query

  ### Module 2
- tesseract and pytesseract was installed and integrated to perform Optical Character Recognition
- To extract text from images pillow was used to open the image
- For the scraping, BeautifulSoup and selenium was used to scrape different products from 'https://www.newegg.com'
- Each product was opened, then the image was downloaded along with their titles then the next page was opened with a delay of 7sec, the images was downloaded for 10 different products then saved in a directory called download_image, which was later renamed to the stockcode in the CNN_Model_Train_Data.csv each served as the label for the cnn model
  ### Module 3
- The stockcode was used as the labels for the data, and a function was made to load and preprocess the data, all images was resized to 128x128 and normalized by dividing the converted RGB by 255. the images and labels was stored
- Since the images was stored in a folder, that folder was uploaded to my drive and can be accessed from from the link above for training the CNN model for image classification.
- The labels was encodedand at the end the label shape was 391x10
- To build the model, tensorflow was used, relu was the activation function for all convolutions and 2 dense layers was made with a softmax activation for the last layer
- The model was compiled using adam optimzer, and for the loss, categorical_crossentropy was used
- The data was split into 80% for training and 20 for testing for 10 epochs
- The model performed with accuracy of 95% and for the validation, 70%
- To handle overfitting, Dropout was used
### Module 4
- A streamlit app was built for 3 main types of query; a text query, an image query, and a product query
- For the text query, the user may type a description of a product and get 10 product suggestions similar to their qury, for the image, the use can upload an image of a text which contains the description of a product, tesseract extracts this text and uses it as input for the query and the user gets 10 similar products as suggestions, then finally for the products, the user may upload a particular product, the CNN model detects which product was uploaded and gives out a label for this products, the labels used to train this model are ids for 10 different products but of course this can be improved later, after identifying the product, it's label which is an id is used to query pinecone for products with similar description and id
 - A flask app for thesame implementation is currently in developement...
