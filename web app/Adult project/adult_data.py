import streamlit as st 

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 


def load_sidebar():
	st.sidebar.header("Income Prediction of a person using given dataset")
	st.sidebar.info('''
						 1) Original owners of database - US Census Bureau\n
						 2) Donor of database - Ronny Kohavi and Barry Becker\n
			        ''')
	st.sidebar.info('''
		  				1) 48842 instances, mix of continuous and discrete (train=32561, test=16281)\n
 						2) 45222 if instances with unknown values are removed (train=30162, test=15060)\n
 					''')

	st.sidebar.warning("not perform Data Modelling Step, So we can predict income now, but I show Data Visualization\
		so we can do Data Analysis")

	st.sidebar.warning("Not Able to write any observations")

def load_dataset():
	df=pd.read_csv("adult.csv",na_values=['?','-','n/a'])
	return df

def drop_unn_data(df):
	st.header("Treatment of missing value and Uneccessary data ")
	df.dropna(axis=0,how='any',inplace=True)
	miss_per=(1-len(df.index)/48842)*100
	st.write("{}% missing value remove from dataset".format(round(miss_per,2)))
	st.write("Two Columns 'fnlwgt','educational-num' remove from datset")
	df.drop(['fnlwgt','educational-num'],axis=1,inplace=True)
	return df

def stat_desc(df):
	st.header("Statistcal Analysis of Data After Removing Missing Values")
	st.subheader("Show Upper 5 row and bottom 5 row data")
	choice=st.radio("Select one option: ",('Top','Bottom'))
	if choice=='Top':
		st.table(df.head())
	else:
		st.table(df.tail())

	st.subheader(" Apply Some Statistcal Functions")
	select=st.selectbox("Select from option below: ",('describe','info','shape','rows','columns'))
	if select=='describe':
		ch=st.radio("describe one of them: ",('numerical','categorial'))
		if ch=='numerical':
			st.write(df.describe(include='number'))
		else:
			st.write(df.describe(include='object'))
	elif select=='info':
		st.write(df.info())
	elif select=='shape':
		st.write(df.shape)
	elif select=='rows':
		st.write("Number of rows in dataset is: ",df.shape[0])
	elif select=='columns':
		st.write("Number of columns in dataset is: ",df.shape[1])

def data_viz(df):
	st.header("Data Visualization")
	choice=st.radio("Types of Analysis: ",("Univariant","Bivariant"))
	if choice=='Univariant':
		select=st.radio("Select any of them: ",('Numerical','Categorial'))
		if select=='Numerical':
			st.text("Histogram For Numerical Data")
			num_df=df.select_dtypes(include='number')
			num_df.hist(figsize=(10,10))
			st.pyplot()

		elif select=='Categorial':			
			st.text("Countplot for categorial Data")
			cat_df=df.select_dtypes(include='object')
			plt.figure(figsize=(12,24))
			plt.subplots_adjust(hspace=1,wspace=1)

			plt.subplot(411)
			sns.countplot(x='workclass', hue='income', data = cat_df)
			plt.title('Income vs Work Class')

			plt.subplot(412)
			sns.countplot(x='occupation', hue='income', data = cat_df)
			plt.xticks(rotation=90)
			plt.title('Income vs Occupation')

			plt.subplot(413)
			sns.countplot(x='education', hue='income', data=cat_df)
			plt.xticks(rotation=90)
			plt.title('Income vs Education')

			plt.subplot(414)
			sns.countplot(x='marital-status', hue='income', data=cat_df)
			plt.xticks(rotation=90)
			plt.title('Income vs Marital Status')

			st.pyplot()

		st.subheader("Observations 1.0")
		if(st.checkbox("Observations of Univariant Analysis: ")):
			st.write('''
				    1) Most of age lie blw 30-50\n
				    2) Majority of Capital gain lie blw 0-10000\n
				    3) Majority of Capital loss lie blw 0-5000\n
				    4) averge working hours per week is in range 35-40\n
				    5) Most of the people are belong to private sector\n
				    6) Most people having salary less than 50K are HS graduate\n
				    7) People who earn income greater than 50K are married-civ-spouse\n
				    ''')

			
				
	elif choice=='Bivariant':
		ch=st.selectbox("Different Representation: ",('Graphical','Tabular'))
		if ch=='Graphical':
			st.text("Income v/s Age")
			sns.boxplot(data=df,x='income',y='age',hue='gender')
			st.pyplot()

			st.text("Income v/s Hours-per-week")
			sns.boxplot(data=df,x='income',y='hours-per-week',hue='gender')
			st.pyplot()

			plt.figure(figsize=(12,12))
			st.text("Age v/s Occupation")
			sns.boxplot(data=df,x='occupation',y='age',hue='income')
			plt.xticks(rotation=90)
			st.pyplot()

		elif ch=='Tabular':

			df['income_category'] = "null"
			df.loc[df['income'] == '>50K', ['income_category']] = 'high income'
			df.loc[df['income'] == '<=50K', ['income_category']] = 'low income'
			st.text("Income v/s Race Pivot Table Representation")
			racewise_income_dist = df.pivot_table(values=['income_category'],index=['income', 'race'],aggfunc = 'count')
			st.table(racewise_income_dist)

			st.text("Income v/s Gender Pivot Table Representation")
			gender_income_dist = df.pivot_table(values=['income_category'],index=['income', 'gender'],aggfunc = 'count')
			st.table(gender_income_dist)

			st.text("Income v/s Relationship Pivot Table Representation")
			rels_income_dist = df.pivot_table(values=['income_category'],index=['income', 'relationship'],aggfunc = 'count')
			st.table(rels_income_dist)

			st.text("Income v/s Occupation Pivot Table Representation")
			occs_income_dist = df.pivot_table(values=['income_category'],index=['income', 'occupation'],aggfunc = 'count')
			st.table(occs_income_dist)


def main():
	st.header("Adult Dataset project: Predict Income of Person using his/her data \n\n")
	load_sidebar()
	df=load_dataset()
	st.write(df.head())
	new_df=drop_unn_data(df)
	stat_desc(new_df)
	data_viz(new_df)

if(__name__=='__main__'):
	main()