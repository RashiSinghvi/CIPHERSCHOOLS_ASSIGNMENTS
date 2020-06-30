import streamlit as st
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import datetime
import seaborn as sns
#from wordcloud import WordCloud,STOPWORDS
from pickle import dump,load
import re
import nltk
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer


def load_sidebar():
	st.sidebar.subheader("Spam Messages Detector")
	st.sidebar.success("Analyze True and Fake messages and detect ads containing messages or fraud messages")
	st.sidebar.info("The collection is composed by just one text file, where each line has the correct class followed by the raw message.")
	st.sidebar.warning("Not perform oversampling/undersampling to overcome problems with imbalanced dataset, So try to text long messages \
		     and give some keywords(urgent,free, prizes) in order to predict spam messages")

@st.cache
def load_dataset():
	df=pd.read_csv("SMSSpamCollection",sep='\t',names=['target','text'])
	return df

def data_desc(df):
	st.header("Datset Description")
	st.subheader("Show Data")
	choice=st.radio("Want to see Top/ Bottom five rows",('Top','Bottom'))
	if choice=='Top':
		st.table(df.head())
	elif choice=='Bottom':
		st.table(df.tail())

	st.subheader(" Apply Some Statistcal Functions")
	select=st.selectbox("Select from option below: ",('describe','shape','rows','columns'))
	if select=='describe':
		st.write(df.describe())
	elif select=='shape':
		st.write(df.shape)
	elif select=='rows':
		st.write("Number of rows in dataset is: ",df.shape[0])
	elif select=='columns':
		st.write("Number of columns in dataset is: ",df.shape[1])

	st.subheader("Message View ")
	mess=st.selectbox("Types of Messages: ",('Ham','Spam'))
	if mess=='Ham':
		ham_df=df.loc[df.target=='ham',:]
		st.write("Total Ham Messages in dataset is: ",ham_df.shape[0])
	elif mess=='Spam':
		spam_df=df.loc[df.target=='spam',:]
		st.write("Total Spam Messages in dataset is: ",spam_df.shape[0])

	st.subheader("Column Names")
	if(st.checkbox("Show columns name in dataset")):
		st.write(df.columns)

def viz(df):
	st.header("Data Visualization")
	st.subheader("Text Analysis")
	choice=st.radio("Choose one of below graph: ",('Bar Graph','Pie Chart'))
	if choice=='Bar Graph':
		plt.title("CountPlot/Bar Graph of Text Messages ")
		st.write(sns.countplot(x='target',data=df))
		st.pyplot()
	else:
		plt.title("Pie Chart of Text Messages ")
		plt.pie(x=df.target.value_counts(),labels=['ham','spam'],explode = [0, 0.1],autopct = '%1.1f%%',shadow=True)
		st.pyplot()

	st.subheader("Observations")
	if(st.checkbox("Creamy observations: ")):
		st.write('''
					1) There are two columns in dataset namely Target(type of message), Text\n
					2) There are approx 86.6%  ham messages and 13.4%  spam messages\n
					3) Also we calculate number of rows in ham or spam messages i.e 4825 and 747 ham and spam \n
					   messages respectively\n
					4) We can say it is imbalanced dataset or they are majority of ham messages\n
				''')
	# st.subheader("WordCloud")
	# typ=st.radio("Select your choice: ",('ham','spam'))


def preprocess2(raw_text):
    # Removing special characters and digits
    letters_only = re.sub("[^a-zA-Z]", " ",raw_text)
    
    # change sentence to lower case
    letters_only = letters_only.lower()

    # tokenize into words
    words = letters_only.split()
    
    # remove stop words                
    words = [w for w in words if not w in stopwords.words("english")]
    
    # Stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    
    clean_sent = " ".join(words)
    
    return clean_sent

def predict(text):
    
    # Loading pretrained CountVectorizer from pickle file
    vectorizer = load(open('pickle/countvectorizer.pkl', 'rb'))
    
    # Loading pretrained logistic classifier from pickle file
    classifier = load(open('pickle/model_lr.pkl', 'rb'))
    
    # Preprocessing the text
    clean_text = preprocess2(text)
    
    # Converting text to numerical vector
    clean_text_encoded = vectorizer.transform([clean_text])
    
    # Converting sparse matrix to dense matrix
    text_input = clean_text_encoded.toarray()
    
    # Prediction
    prediction = classifier.predict(text_input)
    
    return prediction



def main():
	load_sidebar()
	#st.balloons()
	st.header("NLP PROJECT SPAM Messages Detector")
	st.image("spam.jpg", use_column_width = True)
	df=load_dataset()

	#call data_description function
	data_desc(df)

	#call data visualization function
	viz(df)

	#get predicted value
	st.header("Prediction")
	text=st.text_input("Enter your text: ")
	get_pred=predict(text)
	
	if(text):
		if get_pred==1:
			st.write("It is scam message  :angry:")
		else:
			st.write("Relax it is ham message  :grinning:")


if(__name__=='__main__'):
	main()