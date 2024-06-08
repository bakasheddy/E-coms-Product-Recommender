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

  ### Module 2
- tesseract and pytesseract was installed and integrated to perform Optical Character Recognition
- to extract text from two images pillow was used to open the image
- for the scraping, BeautifulSoup and selenium was used to scrape different products from 'https://www.newegg.com'
- each product was opened, then the image was downloaded along with their titles then the next page was opened with a delay of 7sec, the images was downloaded for 10 different products then saved in a directory called download_image, which was later renamed to the stockcode in the CNN_Model_Train_Data.csv each served as the label for the cnn model
  ### Module 3
- the stockcode was used as the labels for the data, and a function was made to load and preprocess the data, all images was resized to 128x128 and normalized by dividing the converted RGB by 255. the images wand labels was stored
- since the images was stored in a folder, that folder was uploaded to my drive and was accessed from there in colab for the training
- the labels was encodedand at the end the label shape was 391x10
- to build the model, tensorflow was used, relu was the activation function on max_pool of 2 and 2 dense layers was made
- the model was compiled using adam optimzer, and for the loss, categorical_crossentropy was used
- the data was split into 80% for training and 20 for testing for 10 epochs
- the model performed with accuracy of 95% and for the validation, 70%
- two images was given and used to make prediction on the model
- to handle overfitting, Dropout was used

## Final note
Thank you for the privilege to go through this vetting test i learnt a lot from it honestly, some of which is my first time using i really enjoyed every step even though i ony did all i could to attempt this, i hope to join you and improve myself better 
