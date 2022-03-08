# Twitter-Sentiment-Analysis on Covid-19 Vaccine (Final Year Project)

The purpose of this project is to gather Covid-19 vaccine tweets from Twitter and perform sentiment analysis on the tweets to develop an understanding on the public's sentiment on Covid-19 vaccine. A model is developed which is a sentiment analysis model to analyze and predict the tweets. The sentiment of the Covid-19 vaccine tweets will be analyzed to determine the sentiment of the public towards Covid-19 vaccine. 

---

- The Jupyter Notebok includes the training of Logistic Regression with TF-IDF, Multinomial Naive Bayes with TF-IDF, and Logistic Regression with Word2Vec.
- The Word2Vec did not perform as well as intended due to the fact of lack of training dataset. As such, Word2Vec is suitable for large corpus data.
- The BERT model fine-tuning is done using Google Colaboratory. The "bert-base-uncased" version of BERT model was used for this project. The number of epochs is 3.
- All models were evaluated using evaluation metrics (f1-score, precision, recall) and confusion matrix.
- The model that was chosen as the sentiment analysis model was the BERT model due to the fact that BERT model is able to predict the sentiment of the tweet with better accuracy. BERT model is also a transformer model that is a strong language model.

---

Twitter Sentiment Analysis on Covid-19 Vaccine App is built using Streamlit. 

>The training dataset can be found from this [link](https://www.kaggle.com/tanujdhiman/twitter-sentiment-analysis/data).

There were attempts of scrapping Covid-19 vaccine tweets directly from Twitter using the **tweepy** python package. However, there were errors in doing such as the ***failed on_status, 429 Too Many Requests*** error. In addition to time constraint, it was deem too time-consuming to solve the error. There is a Kaggle website that hosts a dataset that contains Covid-19 vaccine tweets scraped from Twitter. As such, it was decided that this dataset would be used.

>The Covid-19 vaccine tweet dataset was retrieved from this [link](https://www.kaggle.com/gpreda/all-covid19-vaccines-tweets).

This [website](https://huggingface.co/bert-base-uncased) has more information regarding the "bert-base-uncased" version of BERT model. The BERT model is hosted on huggingface which is a platform provider of machine learning that provides transformer models.

---

**First**, create the conda environment by running the following command in the anaconda command prompt.

```
  conda create --name tweet python=3.8.8
  conda activate tweet
  pip install -r requirements.txt
```  

To run the web application python file, use the command below with **anaconda command prompt** under the right environment. Ensure that the command prompt is in the right directory where the web application python file or specified the directory of the web application python file. 

```
streamlit run Sentiment Analysis on Covid-19 Vaccine Tweet Web App.py
```
