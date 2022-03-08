#%%
# -*- coding: utf-8 -*-
"""
Final Year Project
Twitter Sentiment Analysis on Covid-19 Vaccine Tweets (Web App)

0125118 Lock Jun Lin
"""

#libraries for output (print) input output stream of code
from streamlit.report_thread import REPORT_CONTEXT_ATTR_NAME
from contextlib import contextmanager  #contextlib2 version 0.6.0.post1
from threading import current_thread

#warnings
import warnings
warnings.filterwarnings('ignore')

# utilities
import re
import pandas as pd
import random
import math
import numpy as np
import io 
import sys 
from io import BytesIO

#streamlit version 0.79.0
# import streamlit library for streamlit module to create website application
import streamlit as st
import streamlit.components.v1 as components

#import torch for deep learning model training
import torch
import torch.nn.functional as F #import torch.nn.functional library for calculating prediction probability

#import seaborn and matplotlib library for plotting graph
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import matplotlib.dates as mdates

#import tokenizer for bert to tokenize text 
from transformers import BertTokenizer

from torch.utils.data import TensorDataset #splitting dataset
from transformers_interpret import SequenceClassificationExplainer  #library for explainable AI for transformer model
from torch.utils.data import DataLoader, SequentialSampler #convert dataset into iterable dataset

#wordcloud and spelling check for words
from wordcloud import WordCloud
import wordninja
from spellchecker import SpellChecker
from collections import Counter

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords 

#import keybert for extracting keywords
from keybert import KeyBERT
#import pygooglenews for google news query with extracted keyword
from pygooglenews import GoogleNews

# configure graph style when plotting graph
sns.set(style='darkgrid', palette='muted', font_scale=1.5)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

#%%

#define function to print out code text
@contextmanager
def st_redirect(src, dst):
    placeholder = st.empty()
    output_func = getattr(placeholder, dst)

    with io.StringIO() as buffer:
        old_write = src.write

        def new_write(b):
            if getattr(current_thread(), REPORT_CONTEXT_ATTR_NAME, None):
                buffer.write(b)
                output_func(buffer.getvalue())
            else:
                old_write(b)

        try:
            src.write = new_write
            yield
        finally:
            src.write = old_write

@contextmanager
def st_stdout(dst):
    with st_redirect(sys.stdout, dst):
        yield
        
@st.cache(allow_output_mutation=True, suppress_st_warning=True, max_entries=1)
def init_model():    #function initialize model variables
    #display if there are any available "cuda" device on the device
    if torch.cuda.is_available():    
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    #load tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    
    #import bert model for keyword extraction 
    keyword_model = KeyBERT(model = "princeton-nlp/sup-simcse-bert-base-uncased")

    #load model from destination file path
    model = torch.load('../tweet_bert_model.pt', map_location = device)
    
    return device, tokenizer, model, keyword_model

@st.cache(allow_output_mutation=True, suppress_st_warning=True, max_entries=1)
def init_data():    #function to initialize data variable
    #load covid-19 vaccine tweet with predicted sentiment dataset
    df_vax = pd.read_csv('labelled_predict_vax_tweet.csv')
    df_vax.sentiment = df_vax.sentiment.astype('category')  #convert 'Sentiment' column into categorical value type

    #add 'sentiment_num' column and convert into integer data type
    df_vax['sentiment_num'] = df_vax['sentiment'].map({'negative':-1,'neutral':0,'positive':1})
    df_vax.sentiment_num = df_vax.sentiment_num.astype('int32')
    
    # Convert string to a list of words and insert into new column in dataframe
    df_vax['words'] = df_vax.text.apply(lambda x:re.findall(r'\w+', x ))
    
    #array of all vaccine type and country
    all_vax = ['covaxin', 'sinopharm', 'sinovac', 'moderna', 'pfizer', 'biontech', 'oxford', 'astrazeneca', 'sputnik']
    countries = ['india','usa','canada','spain','uk','brazil', 'australia', 'japan', 'malaysia']
    
    #dataframe to store mean sentiment of corresponding tweet based on user location
    c_sentiment=pd.DataFrame()
    c_sentiment['countries']=countries
    c_senti=list()
    for c in countries :
        c_senti.append(df_vax[df_vax['loc_label'] == c].sentiment_num.mean())
    c_sentiment['sentiment']=c_senti
    
    #dataframe to store vaccine type and mean sentiment of corresponding vaccine tweet of vaccine type
    vax_sentiment = pd.DataFrame()
    vax_sentiment['vaccine']=all_vax
    v_senti = list()
    for v in all_vax :
        v_senti.append(df_vax[df_vax['vax_type'] == v].sentiment_num.mean())
    vax_sentiment['sentiment']=v_senti
    
    #get stop words list to remove stop words from tweet 
    stop_words = set(stopwords.words('english'))  
    stop_words.add("amp")
    
    #class name of sentiment
    class_names = ['negative', 'neutral', 'positive']

    return df_vax, stop_words, all_vax, countries, vax_sentiment, c_sentiment, class_names

#%%

#function to predict tweet and display graph and explainable AI
def tweet_pred(tweet_input):
    #array to store input id and attention mask from tokenizer
    pred_input_ids = []
    pred_attention_masks = []

    #encode the tweet input into input id and attention mask    
    encoded_dict = tokenizer.encode_plus(
                        tweet_input,                      
                        add_special_tokens = True, 
                        max_length = 315,           
                        pad_to_max_length = True,
                        return_attention_mask = True,   
                        return_tensors = 'pt',     
                    )
      
    pred_input_ids.append(encoded_dict['input_ids'])
    pred_attention_masks.append(encoded_dict['attention_mask'])
    
    pred_input_ids = torch.cat(pred_input_ids, dim=0)
    pred_attention_masks = torch.cat(pred_attention_masks, dim=0)
    
    #convert encoded tweet input into iterable dataset with defined batch size
    tweet_input_load = TensorDataset(pred_input_ids, pred_attention_masks)
    batch_size = 32
    vax_dataloader = DataLoader(
                tweet_input_load,  
                sampler = SequentialSampler(tweet_input_load),     #Sequential Sampling
                batch_size = batch_size 
            )
    
    #perform prediction on tweet input
    predictions = []  #array to store prediction probability of tweet input
    for batch in vax_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask = batch
      
        with torch.no_grad():
            pred_result = model(b_input_ids, 
                           token_type_ids=None, 
                           attention_mask=b_input_mask,
                           return_dict=True)
      
        logits = pred_result.logits
        logits = logits.detach().cpu().numpy()
        logits=np.argmax(logits,axis=1)
        for i in range(len(logits)):
            predictions.append(logits[i])
    
    # retrieve the probability prediction of the tweet input by the model
    proba = F.softmax(pred_result.logits, dim=1)[0].tolist()
    
    class_names = ['negative', 'neutral', 'positive']
    #display side bar graph with prediction probability of tweet input
    pred_df = pd.DataFrame({
      'class_names': class_names,
      'values': proba })
    st.subheader("Sentiment Prediction")
    st.markdown("Bar Graph of Sentiment Prediction Probability (Confidence)")
    st.markdown("#### Input Tweet: %s" % tweet_input)
    fig = plt.figure(figsize = (10,5))
    sns.barplot(x='values', y='class_names', data=pred_df, orient='h')
    #value of probability prediction (converted into percentage)
    plt.text(0, 0, r'${0:2.2f}\%%$'.format(proba[0]*100))
    plt.text(0, 1, r'${0:2.2f}\%%$'.format(proba[1]*100))
    plt.text(0, 2, r'${0:2.2f}\%%$'.format(proba[2]*100))
    plt.title('Sentiment Prediction of Tweet Input (Confidence)')
    plt.ylabel('sentiment')
    plt.xlabel('probability (confidence)')
    plt.xlim([0, 1])
    st.pyplot(fig)
    
    #display explainable AI on prediction of input tweet by the model
    cls_explainer = SequenceClassificationExplainer(model, tokenizer)
    attributions = cls_explainer(tweet_input, internal_batch_size=2)
    
    #replace numerical label with corresponding sentiment label
    if cls_explainer.predicted_class_name == 'LABEL_0':
      st.markdown("### **Predicted Sentiment: Negative (Label_0)** \n")
    elif cls_explainer.predicted_class_name == 'LABEL_1':
      st.markdown("### **Predicted Sentiment: Neutral (Label_1)**\n")
    elif cls_explainer.predicted_class_name == 'LABEL_2':
      st.markdown("### **Predicted Sentiment: Positive (Label_2)**\n")
    
    #display explainable AI as HTML component
    st.subheader("Explainable AI")
    if attributions:
        word_attributions_expander = st.beta_expander("Click here for raw word attributions...")
        with word_attributions_expander:
            st.json(attributions)
        components.html(cls_explainer.visualize()._repr_html_(), scrolling=True, height=350)

#function to extract keyword and perform google news query
def keyword_extraction(tweet_input): 
    #extract and display keywords
    k_words = keyword_model.extract_keywords(tweet_input, top_n=2,use_mmr=True, 
                                             keyphrase_ngram_range=(1,2), stop_words="english")
    st.write("Keywords Extracted:-\n")
    #retrieve the string of the result of the keyword extraction
    words_Str = ' '.join([str(elem) for elem in k_words])
    syntax = re.compile("'[^']*'")
    for value in syntax.findall(words_Str):
        st.markdown("""
                    - **%s**
                    """ % value)
    k_words_expander = st.beta_expander("Click here for raw word attributions of extracted keyword...")
    with k_words_expander:
        st.json(k_words)

    #google news query with extracted keywords for english langugage news only
    gn = GoogleNews(lang = 'en')
    st.write("Google News Query Results:-")
    for value in syntax.findall(words_Str):
        search = gn.search(value)
        newsitem = search['entries']
        #display google news query results
        st.markdown("""
                    >**Extracted word**: **%s**
                    >
                    >**Article/News Title**: %s
                    >
                    >**Article/News Link**: %s""" %(value, newsitem[0].title, newsitem[0].link))

#%%

#function to filter dataframe
def filter_data(df, vax=None, country=None):
    #retrieve sentiment of corresponding tweet of vaccine type or country passed into function
    label = "all"
    if vax != None:
        df = df[df['vax_type'] == vax]
        label = vax
    if country != None:
        df = df[df['loc_label'] == country]
        label = country
    return df, label    

#function to plot distribution pie chart
def distribution_graph(df, label):
    #plot pie chart
    fig = plt.figure(figsize = (12,7))
    df['sentiment'].value_counts(normalize=True).plot.pie(autopct='%1.1f%%', shadow=True,
                                                              startangle=90, explode = (0.1,0.1,0.1))
    plt.xticks(rotation=0)
    plt.ylabel("")
    plt.title(f'Distribution of Covid-19 Vaccine Tweet (%s)' % label.capitalize())
    #display pie chart as picture
    buf = BytesIO()
    fig.savefig(buf, format="png")
    st.image(buf)
    
#function to plot line graph (sentiment variance)
def time_variance(df, label):
    #temporary array to store mean sentiment of each date
    temp=pd.DataFrame()
    temp['date'] = sorted(df['date'].unique())
    senti=list()
    for date in temp['date']:
        senti.append(df[df['date']==date].sentiment_num.mean())
    temp['sentiment']=senti
    
    #plot line graph
    fig, ax = plt.subplots(figsize=(14,7))
    sns.lineplot(ax=ax,x='date',y='sentiment', data= temp)
    #plot x-axis(date) as monthly date
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.set_xlabel("Time (Monthly Date)",size=16)
    ax.set_ylabel("Sentiment",size=16)
    ax.set_title(f"Variance of Sentiment with regard to Time (%s)" % label.capitalize(), size=20)
    plt.grid(True)
    st.pyplot(fig)
    
# Function to filter the data to a single vaccine and plot the timeline
# Note: a lot of the tweets seem to contain hashtags for multiple vaccines even though they are specifically referring to one vaccine - not very helpful!
def filtered_timeline(df, label):    
    # Get counts of number of tweets by sentiment for each date
    timeline = df.groupby(['date', 'sentiment']).agg(**{'tweets': ('id', 'count')}).reset_index()
    #plot timeline graph
    fig = px.line(timeline, x='date', y='tweets', color='sentiment', 
                  category_orders={'sentiment': ['neutral', 'negative', 'positive']},
                  title=f'Timeline showing sentiment of Covid-19 vaccine tweets (%s)' % label.capitalize())
    st.plotly_chart(fig)
    
#function to plot bar graph of comparison of mean sentiment
def mean_senti_plot(v_df, c_df):
    #plot bar chart
    fg, axs=plt.subplots(figsize=(14,7))
    sns.barplot(ax=axs, x='vaccine', y='sentiment', data=v_df)
    axs.set_xlabel("Vaccines",size=16)
    axs.set_ylabel("Mean Sentiment",size=16)
    axs.set_title("Comparison of Mean Sentiment of Vaccines",size=20)
    axs.grid(True)
    st.pyplot(fg)

    #plot bar chart
    fg, axs=plt.subplots(figsize=(15,8))
    sns.barplot(ax=axs,x='countries',y='sentiment',data=c_df)
    axs.set_xlabel("Countries",size=16)
    axs.set_ylabel("Mean Sentiment",size=16)
    axs.set_title("Comparison of Mean Sentiment of Countries",size=20)
    axs.grid(True)
    st.pyplot(fg)

#%%

# FUNCTIONS REQUIRED
def flatten_list(l):
    return [x for y in l for x in y]

def is_acceptable(word: str):
    return word not in stop_words and len(word) > 2

# Color coding our wordclouds 
def red_color_func(word, font_size, position, orientation, random_state=None,**kwargs):
    return f"hsl(0, 100%, {random.randint(25, 75)}%)" 

def green_color_func(word, font_size, position, orientation, random_state=None,**kwargs):
    return f"hsl({random.randint(90, 150)}, 100%, 30%)" 

def yellow_color_func(word, font_size, position, orientation, random_state=None,**kwargs):
    return f"hsl(42, 100%, {random.randint(25, 50)}%)" 

# Reusable function to generate word clouds 
def generate_word_clouds(neg_doc, neu_doc, pos_doc, label):
    # Display the generated image (wordcloud):
    fig, axes = plt.subplots(3,1, figsize=(15,8))
    
    wordcloud_neg = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(" ".join(neg_doc))
    axes[0].imshow(wordcloud_neg.recolor(color_func=red_color_func, random_state=3), interpolation='bilinear')
    axes[0].set_title(f"Most Frequent Words of Negative Tweets (%s)" % label.capitalize(), fontsize = 16)
    axes[0].axis("off")

    wordcloud_neu = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(" ".join(neu_doc))
    axes[1].imshow(wordcloud_neu.recolor(color_func=yellow_color_func, random_state=3), interpolation='bilinear')
    axes[1].set_title("Most Frequent Words of Neutral Tweets (%s)" % label.capitalize(), fontsize = 16)
    axes[1].axis("off")

    wordcloud_pos = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(" ".join(pos_doc))
    axes[2].imshow(wordcloud_pos.recolor(color_func=green_color_func, random_state=3), interpolation='bilinear')
    axes[2].set_title("Most Frequent Words of Positive Tweets (%s)" % label.capitalize(), fontsize = 16)
    axes[2].axis("off")

    plt.tight_layout()
    st.pyplot(fig);

def get_top_percent_words(doc, percent):
    # Returns a list of "top-n" most frequent words in a list 
    top_n = int(percent * len(set(doc)))
    counter = Counter(doc).most_common(top_n)
    top_n_words = [x[0] for x in counter]
    
    return top_n_words
    
def clean_document(doc):
    spell = SpellChecker()
    lemmatizer = WordNetLemmatizer()
    
    # Lemmatize words (needed for calculating frequencies correctly )
    doc = [lemmatizer.lemmatize(x) for x in doc]
    
    # Get the top 10% of all words. This may include "misspelled" words 
    top_n_words = get_top_percent_words(doc, 0.1)

    # Get a list of misspelled words 
    misspelled = spell.unknown(doc)
    
    # Accept the correctly spelled words and top_n words 
    clean_words = [x for x in doc if x not in misspelled or x in top_n_words]
    
    # Try to split the misspelled words to generate good words (ex. "lifeisstrange" -> ["life", "is", "strange"])
    words_to_split = [x for x in doc if x in misspelled and x not in top_n_words]
    split_words = flatten_list([wordninja.split(x) for x in words_to_split])
    
    # Some splits may be nonsensical, so reject them ("llouis" -> ['ll', 'ou', "is"])
    clean_words.extend(spell.known(split_words))
    
    return clean_words

def get_log_likelihood(doc1, doc2):    
    doc1_counts = Counter(doc1)
    doc1_freq = {
        x: doc1_counts[x]/len(doc1)
        for x in doc1_counts
    }
    
    doc2_counts = Counter(doc2)
    doc2_freq = {
        x: doc2_counts[x]/len(doc2)
        for x in doc2_counts
    }
    
    doc_ratios = {
        # 1 is added to prevent division by 0
        x: math.log((doc1_freq[x] +1 )/(doc2_freq[x]+1))
        for x in doc1_freq if x in doc2_freq
    }
    
    top_ratios = Counter(doc_ratios).most_common()
    top_percent = int(0.1 * len(top_ratios))
    return top_ratios[:top_percent]

# Function to generate a document based on likelihood values for words 
def get_scaled_list(log_list):
    counts = [int(x[1]*100000) for x in log_list]
    words = [x[0] for x in log_list]
    cloud = []
    for i, word in enumerate(words):
        cloud.extend([word]*counts[i])
    # Shuffle to make it more "real"
    random.shuffle(cloud)
    return cloud

#function to calculate and plot word cloud
def get_smart_clouds(df, label):
    #retrieve word of corresponding tweet
    neg_doc = flatten_list(df[df['sentiment']=='negative']['words'])
    neg_doc = [x for x in neg_doc if is_acceptable(x)]

    pos_doc = flatten_list(df[df['sentiment']=='positive']['words'])
    pos_doc = [x for x in pos_doc if is_acceptable(x)]

    neu_doc = flatten_list(df[df['sentiment']=='neutral']['words'])
    neu_doc = [x for x in neu_doc if is_acceptable(x)]

    # Clean all the documents
    neg_doc_clean = clean_document(neg_doc)
    neu_doc_clean = clean_document(neu_doc)
    pos_doc_clean = clean_document(pos_doc)

    # Combine classes B and C to compare against A (ex. "positive" vs "non-positive")
    top_neg_words = get_log_likelihood(neg_doc_clean, flatten_list([pos_doc_clean, neu_doc_clean]))
    top_neu_words = get_log_likelihood(neu_doc_clean, flatten_list([pos_doc_clean, neg_doc_clean]))
    top_pos_words = get_log_likelihood(pos_doc_clean, flatten_list([neu_doc_clean, neg_doc_clean]))

    # Generate syntetic a corpus using our loglikelihood values 
    neg_doc_final = get_scaled_list(top_neg_words)
    neu_doc_final = get_scaled_list(top_neu_words)
    pos_doc_final = get_scaled_list(top_pos_words)

    # Visualise our synthetic corpus
    generate_word_clouds(neg_doc_final, neu_doc_final, pos_doc_final, label)

#%%

# title of the application
st.title('Twitter Sentiment Analysis on Covid-19 Vaccine Web App')

# set layout for side bar
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 350px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
        margin-left: -350px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# set sidebar content
st.sidebar.title('Twitter Sentiment Analysis on Covid-19 Vaccine Web App')
st.sidebar.subheader('Webpages')

# set drop down select box for user to navigate different page of application
app_mode = st.sidebar.selectbox('Choose Webpage', [
                                'About App', 'Sentiment Prediction for Covid-19 Vaccine Tweet and Keyword Extraction', "Dashboard", "Model Information"])
        
#%%

#initialize model
device, tokenizer, model, keyword_model = init_model()

#initialize dataframe
df_vax, stop_words, all_vax, countries, vax_sentiment, c_sentiment, class_names= init_data()

country = None 
vax = None

#%%

if app_mode == 'About App':
    st.sidebar.markdown("---")
    st.image('image/sentiment_analysis_pic.jpeg')
    st.header('About Twitter Sentiment Analysis on Covid-19 Vaccine Application')
    st.markdown('''
                This is a **Twitter Sentiment Analysis on Covid-19 Vaccine Application**.
                
                The application takes in a tweet that is entered manually by the user into the system and classify the tweet using the Twitter sentiment analysis model. 
                
                The sentiment analysis model is trained using Google Colab and is exported into a **Panther Project Format files** to be loaded later by this application. The tokenizer that 
                will be used for tokenizing the text data will be the “bert-base-uncased” version of BertTokenizer. As such, the training of the sentiment analysis model has already 
                been completed.
                
                BERT Model which is a Transformer (deep learning) model is used for developing the Sentiment Analysis Model. A pre-trained BERT model is applied and then fine-tuned 
                using the tweet dataset which is the dataset to be used for fine tuning the BERT model in order to allow the model to perform sentiment analysis on tweets. 

                The BERT model that will be used will be the __"bert-base-uncased"__ version of BERT model that is a pre-trained BERT model from Hugging Face which is the 
                transformers python library that provides thousands of pretrained models to perform tasks on different modalities such as text, vision, and audio. The tokenizer that 
                will be used for tokenizing the text data will be the “bert-base-uncased” version of BertTokenizer.

                The “bert-base-uncased” version of BERT, which is the smaller model trained on lower-cased English text (with 12-layer, 768-hidden, 12-heads, 110M parameters). 
                The following [website](https://huggingface.co/bert-base-uncased) is the official website with information regarding the “bert-base-uncased” version of BERT.
                
                The dataset that was used to train the sentiment analysis model, *twitter_training.csv* was downloaded from Kaggle. Click 
                [here](https://www.kaggle.com/tanujdhiman/twitter-sentiment-analysis/data) to go to the Kaggle website hosting the dataset.

                There is a **Dashboard** page that displays the result of the sentiment analysis of Covid-19 vaccine tweets retrieved from Twitter. The **Dashboard** page displays
                the result in the form of **Graphs** and **WordCloud** to visualize the result which can be easily interpreted by the user. The Covid-19 vaccine tweets is a dataset where the 
                Covid-19 vaccine tweets is scraped from Twitter using the *tweepy* Python package. The **Covid-19 All Vaccines Tweet** dataset can be downloaded from this
                [link](https://www.kaggle.com/gpreda/all-covid19-vaccines-tweets).
                
                There is a page that displays the evaluation metrics of the Twitter sentiment analysis model called **"Model Information"** as well.
                ''')
    
elif app_mode == 'Sentiment Prediction for Covid-19 Vaccine Tweet and Keyword Extraction':
    st.sidebar.markdown("---")
    st.header("Sentiment Prediction for Covid-19 Vaccine Tweet")
    st.markdown("""
                ### User enter Covid-19 Vaccine tweet _(or any tweets)_  to predict the sentiment of the tweet.
                
                _User enter the tweet into the text input field below._
                """)
    try:            
        #receive tweet input from user
        tweet_input = st.text_area("Input tweet here :", "I like you, I love you", height= 50)
    #error message if an error has occur            
    except ValueError:
        st.error("An error has occured. Please enter a valid input.")
    
    if tweet_input == "":
        st.error("Error: Did not receive tweet input. Please input a tweet.")

    # tweet sentiment analysis will run when user has entered a tweet
    if tweet_input != "":
        tweet_pred(tweet_input)
        #remove url in tweet input 
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        tweet_input_clean = url_pattern.sub(r'', tweet_input)
        st.subheader("Keyword Extraction and Google News Query With Extracted Keyword")
        keyword_extraction(tweet_input_clean)

elif app_mode == "Dashboard":
    st.markdown("""
                ## Dashboard
                
                This webpage displays the result of the sentiment analysis of the Covid-19 vaccine tweets retrieved from Twitter. 
                The graphs shown below is the visualization of the result of the sentiment analysis. 
                """)
    st.sidebar.markdown('---')    
    
    #user choose to filter result according to vaccine type or country
    filter_option = st.sidebar.radio("Choose Filter:-", ['All', 'Vaccine Type', 'Country'])
    st.sidebar.markdown('---')
    if filter_option == "Vaccine Type":
        vax = st.sidebar.selectbox("Choose Vaccine Type:- ", all_vax)
        st.sidebar.markdown("---")
    elif filter_option == "Country":
        country = st.sidebar.selectbox("Choose Country:- ", countries)    
        st.sidebar.markdown("---")
    filtered_df_vax, label = filter_data(df_vax, vax, country)
    #display all graph
    st.subheader("Distribution of Target Variable (sentiment)")
    distribution_graph(filtered_df_vax, label)
    st.subheader("Line Graph of Variance of Sentiment with regard to Time")
    time_variance(filtered_df_vax, label)
    st.subheader("Timeline Graph for Showing Sentiment of Covid-19 vaccine tweets")
    filtered_timeline(filtered_df_vax, label)
    if filter_option == 'All':
        st.subheader("Bar Graph of Comparison of Mean Sentiment of Vaccines/Countries")
        mean_senti_plot(vax_sentiment, c_sentiment)
    st.subheader("WordCloud to Visualize Most Frequent Words in Covid-19 Vaccine Tweets of Respective Sentiment")
    get_smart_clouds(filtered_df_vax, label)

elif app_mode == "Model Information":
    st.sidebar.markdown("---")
    st.header("Model Information (BERT model) (bert-base-uncased)")
    with st.beta_expander("Click to see model layer information..."):
        with st_stdout("code"):
              print(model)
    st.subheader("Classification Report (Evaluation Metrics of Model)")
    st.image("image/classification_report.png")
    st.subheader("Confusion Matrix of Model")
    st.image('image/confusion_matrix.png')
    st.subheader('Training/Validation Loss Curve')
    st.image('image/train_valid_loss.png')
    