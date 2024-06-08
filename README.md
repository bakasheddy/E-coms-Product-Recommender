# test

link to the scraped data is [here](https://drive.google.com/drive/folders/1F4YVmGjnaF63L6JrsYvwAXVaeHd7ej5L?usp=drive_link)

### module 1
- the data was loaded using pandas on google colab and local jupyter notebook and each feature was thoroughly examined for anomalys
- regular expression was utilzedd for the cleaning on features which had repeated patterns of untidiness
- features was converted to there appropriate data types
- duplicates and NAN values that could not be helped was dropped and a clean_data was downloaded
  
- pinecone was installed and access was granted using my api key
- nltk was used to tokenize the 'Description' column and word2vec was used to create a word2vwc model from tokens generated
- using the word2vec model a vector representations for each product description was created, the average the word vectors was taken and cases where no vectors were found was handled
- the 'StockCode' feature was used as the products ids and a pincone index was created withe the name productvectors and cosine similarity was used for the matric because It efficiently captures the similarity between vectors while mitigating the effects of varying vector magnitudes.
- the index was connected and a function was made to preprocess a user's query
- fuction was made to recommend products similar to the user's query
- for some reason not known to me at this time this function does not work like it should but i hope to improve the code and make it work as intended
  
- tesseract and pytesseract was installed and integrated to perform Optical Character Recognition
- to extract text from two images pillow was used to open the image

  ### Module 2
  - 
