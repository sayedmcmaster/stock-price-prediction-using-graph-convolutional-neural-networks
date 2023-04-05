#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[1]:


import pandas as pd
# Import Libraries for Graph, GNN, and GCN

import stellargraph as sg
from stellargraph import StellarGraph

from stellargraph.mapper import FullBatchNodeGenerator
from stellargraph.layer import GCN


# In[2]:


# Machine Learnig related library Imports

from tensorflow.keras import layers, optimizers, losses, metrics, Model
from sklearn import preprocessing, model_selection
from IPython.display import display, HTML
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# was active

data_folder = './data/yahoonewsarchive/'
# os.chdir(data_folder);
# file = data_folder + 'NEWS_YAHOO_stock_prediction.csv';
file = data_folder + 'News_Yahoo_stock.csv';


# In[4]:


df_news = pd.read_csv(file)
df_news.head()


# In[5]:


df_news = df_news[:100]


# # Approaches: Find all stock tickers in an/all article/articles
# 
# 1. Find code that does this: from internet or from previous work or from courses that you have taken online or in academia
# 2. Iterative read the article and match with stock tickers, and find all tickers. Drawback: to which tickers to match or how will you know what is a ticker? Any two to four letters Uppercase, NASDAQ AAPL
# 3. Load the article in database and then use SQL -> may not work that well unless you write some functions
# 4. NLTK, remove stop words, find all tokens, then find All Uppercase words. create a list. attach article ids to the list. Then match with the list of tockers. find common tickers between them. then create tuples with two (indicating edge) (source target weight) 

# In[6]:


# import NLTK libraries
# remove stop words using NLTK methods 
# remove all sorts of unnecessary words
# find all tokens
# Keep only All Uppercase words in a list : dictionary/map: dataframe will be ideal
# create a list/dictionary/map: dataframe will be ideal. attach article ids to the list/dataframe data.
# Create a list of all NasDAQ Tickers
# Then match with the list of NASDAQ tockers. 
# find common tickers between them. 
# then create tuples with two (indicating edge) (source target weight)
# increase weight for each article and pair when you see a match


# In[7]:


# import NLTK libraries
import nltk


# In[8]:


# remove stop words using NLTK methods 
# remove all sorts of unnecessary words
# find all tokens
# Keep only All Uppercase words in a list : dictionary/map: dataframe will be ideal

from nltk.tokenize import RegexpTokenizer

dataFrameWithOnlyCapitalWords = pd.DataFrame(columns =  ["id", "Title", "Content"]) 
for index, row in df_news.iterrows():
    # print(row[id], row['title'], row['content'])
                
    # words with capital letters in the beginning +  as much as possible
    capitalWords = RegexpTokenizer('[A-Z]+[A-Z]\w+')
    # print("\n::All Capital Words::", capitalWords.tokenize(row['content']))
    allCapitalWords = capitalWords.tokenize(row['content'])
        
    dataFrameWithOnlyCapitalWords.loc[index] = [index, row['title'], allCapitalWords]
    #break



dataFrameWithOnlyCapitalWords.head() #, dataFrameWithOnlyCapitalWords.shape


# # Create a list of all (NasDAQ) 30 stocks as per the paper
# 

# In[9]:


# Find/Create a list of NASDAQ Stocks
import os
import glob
nasdaqDataFolder = './archive/stock_market_data/nasdaq/csv'
os.chdir(nasdaqDataFolder)





# In[10]:


# Create a list of all NasDAQ Tickers

extension = "csv"
fileTypesToMerge = ""
# all_filenames = [i for i in glob.glob('*' + '*.{}'.format(extension))]
all_nasdaq_tickers = [i[:-4] for i in glob.glob('*' + fileTypesToMerge + '*.{}'.format(extension))]
nasdaq_tickers_to_process = all_nasdaq_tickers #[:10]
nasdaq_tickers_to_process


# In[11]:


nasdaq_tickers_to_process.remove('FREE')
nasdaq_tickers_to_process.remove('CBOE')
nasdaq_tickers_to_process.remove('III')
nasdaq_tickers_to_process.remove('RVNC')
sorted(nasdaq_tickers_to_process)


# In[ ]:





# In[12]:


fortune_30_tickers_to_process = [
'WMT',
'XOM',
'AAPL',
'UNH',
'MCK',
'CVS',
'AMZN',
'T',
'GM',
'F',
'ABC',
'CVX',
'CAH',
'COST',
'VZ',
'KR',
'GE',
'WBA',
'JPM',
'GOOGL',
'HD',
'BAC',
'WFC',
'BA',
'PSX',
'ANTM',
'MSFT',
'UNP',
'PCAR',
'DWDP']




# In[ ]:





# nasdaq_tickers_to_process = [
# 'WMT',
# 'XOM',
# 'AAPL',
# 'UNH',
# 'MCK',
# 'CVS',
# 'AMZN',
# 'T',
# 'GM',
# 'F',
# 'ABC',
# 'CVX',
# 'CAH',
# 'COST',
# 'VZ',
# 'KR',
# 'GE',
# 'WBA',
# # 'JPM',
# #'GOOGL',
# 'HD',
# 'BAC',
# 'WFC',
# 'BA',
# 'PSX',
# 'ANTM',
# 'MSFT',
# 'UNP',
# 'PCAR',
# 'DWDP']
# 

# # Find NASDQ Tickers in each article
# Create graph steps
# Find all edges 

# In[13]:


combinedTupleList = [];
allMatchingTickers = [];
from itertools import combinations
for index, row in dataFrameWithOnlyCapitalWords.iterrows():
    #print(index)
    #print(set(row['Content']))
    #print(set(nasdaq_tickers_to_process));    
    matchingTickers = set(set(fortune_30_tickers_to_process).intersection(set(row['Content'])))
    #print(matchingTickers)
    if len (matchingTickers) > 1:
        allTuples = list(combinations(matchingTickers, 2));
        #print(list(combinations(matchingTickers, 2)))
        
        #allMatchingTickers = set(allMatchingTickers).union(matchingTickers);
        for aTuple in allTuples:
            combinedTupleList.append(tuple(sorted(aTuple)));
            allMatchingTickers.append(aTuple[0])
            allMatchingTickers.append(aTuple[1])
            
        
    # print("*******************");
    #break
    
#combinedTupleList = list(set(combinedTupleList))
allMatchingTickers = set(allMatchingTickers)

# list(set(combinedTupleList)), len(combinedTupleList), len(set(combinedTupleList)), allMatchingTickers, len(allMatchingTickers), len(set(allMatchingTickers))
sorted(combinedTupleList), type(aTuple), type(sorted(aTuple)), allMatchingTickers


# In[14]:


#combinedTupleList[:1], set(allMatchingTickers)


# In[15]:


# calculate edge weights
from collections import Counter

tuplesWithCount = dict(Counter(combinedTupleList))
tuplesWithCount


# In[16]:


l = list(tuplesWithCount.keys())
l

#print(list(zip(*l))[0])
#print(list(zip(*l))[1])

source = list(zip(*l))[0];
target = list(zip(*l))[1];
edge_weights = tuplesWithCount.values()
source, target, edge_weights, len(source), len(target)


# In[62]:


import networkx as nx
Graph_news = nx.Graph(tuplesWithCount.keys())
nx.draw_networkx(Graph_news, pos = nx.circular_layout(Graph_news), node_color = 'r', edge_color = 'b')
#tuplesWithCount.keys()


# # Finally Create graph based on financial news

# In[17]:


import os
os.getcwd()
os.chdir('../../../../')
#os.chdir('./mcmaster/meng/747/project/')
os.getcwd()


# In[18]:


# Now create node data i.e time series to pass as part of the nodes
'''
df = pd.DataFrame();
data_file = "../../../..//archive/stock_market_data/nasdaq/nasdq-stock-price--all-merged.csv"
# stock-price--all-merged.csv"
df = pd.read_csv(data_file);
df.head()
'''

# this is the place where the new dataset starts i.e. fortune 30 companies
df = pd.DataFrame();
data_file = "per-day-fortune-30-company-stock-price-data.csv";
df = pd.read_csv("./data/" + data_file, low_memory = False);
df.head()


# In[19]:


df.index


# In[20]:


drop_cols_with_na = 1
drop_rows_with_na = 0


# In[21]:


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.interpolate.html


try:
  df = df.interpolate(inplace = False)
except:
  print("An exception occurred. Operation ignored")
  exit
    
df.isnull().values.any()
df[df.isna().any(axis = 1)]  


#----



if drop_cols_with_na == 1:
    df = df.dropna(axis = 1);    
   
df, df.shape

## -- 

df.isnull().values.any()
df[df.isna().any( axis = 1 )]


## --

# df_s_transpose.index = df_s_transpose['Date']
#df.index = df.index.astype('datetime64[ns]')
df


# In[22]:


df_s =  df #[ ['Ticker', 'Date', 'Adjusted Close'] ];
df_s


# In[23]:


df_s["Date"] = df_s["Date"].astype('datetime64[ns]')
df_s = df_s.sort_values( by = 'Date', ascending = True )
df_s


# df_s_pivot = df_s.pivot_table(index = 'Ticker', columns = 'Date', values = 'Adjusted Close')
# df_s_pivot

# In[24]:


allMatchingTickers


# 
# 
# drop_rows_with_na = 0
# if drop_rows_with_na == 1:
#     df_s_transpose = df_s_transpose.dropna(axis=0);
#     #df_s_transpose["Date"] = df_s_transpose["Date"].astype('datetime64[ns]')
#     #df_s_transpose.sort_values(by='Date', ascending=False)
#     df_s_transpose.to_csv('../../../..//archive/stock_market_data/nasdaq/-na-dropped-nasdq-stock-price--all-merged.csv');
#    
# df_s_transpose.head(100)
# 
# 

# In[25]:


df_s_transpose = df_s #_pivot.T
df_s_transpose

df_s_transpose_feature = df_s_transpose.reset_index(drop = True, inplace=False)
# df_s_transpose_feature =  df_s_transpose_feature.values.tolist()
# print(df_s_transpose_feature.values.tolist())
#df_s_transpose_feature['AAPL'].values



# In[26]:


df_s_transpose_feature = df_s_transpose.set_index('Date')


# In[27]:


df_s_transpose_feature


# In[28]:


#df_s_transpose['SIMO']


# In[29]:


# df_s_transpose_feature['AAPL'].values
len(allMatchingTickers), len(set(allMatchingTickers)) #, df_s['Ticker']
#df_s_tickers = df_s['Ticker'];
#len(df_s_tickers), len(set(allMatchingTickers)), df_s_transpose.columns.unique, len(set(df_s['Ticker']))
#df_s_tickers = list(set(df_s['Ticker'])); # list(df_s_transpose.columns.unique) #
#sorted(df_s_tickers)
#for x in df_s_tickers:
 #   print(x)


# In[30]:


df_s_tickers = df_s_transpose_feature.columns
#df_s_tickers = list(set(df_s_tickers.drop('Date')))
sorted(df_s_tickers[:5])


# In[31]:


set_allMatchingTickers = set(allMatchingTickers)
df_s_tickers = fortune_30_tickers_to_process #list(set(df_s['Ticker'])); # list(df_s_transpose.columns.unique) #
node_Data_financial_news = [];

'''
for x in set_allMatchingTickers :
    # if x in df_s_tickers:
    print(x)
    node_Data_financial_news.append( df_s_transpose_feature[x].values)
'''  

node_Data_financial_news = pd.DataFrame(df_s_transpose_feature) #, index = list(allMatchingTickers)) #, index = list(set_allMatchingTickers))
#node_Data_financial_news = node_Data_financial_news.T 
node_Data_financial_news


# In[32]:


node_Data_financial_news = node_Data_financial_news.T
node_Data_financial_news


# In[33]:


node_Data_financial_news


# node_Data_financial_news#.drop(axis = 0)
# node_Data_financial_news = node_Data_financial_news.T
# node_Data_financial_news

# node_Data_financial_news = node_Data_financial_news.drop('Date')
# node_Data_financial_news

# In[34]:


financial_news_edge_data = pd.DataFrame(
    {"source": source, "target": target, "edge_feature": edge_weights}
)

financial_news_graph = StellarGraph(node_Data_financial_news, edges = financial_news_edge_data, node_type_default="corner", edge_type_default="line")
print(financial_news_graph.info())


# In[35]:


# debug code
# financial_news_graph_data,  sorted(node_Data_financial_news.columns.unique())
# [1,2] + [2, 3,4], set(source + target).difference(sorted(node_Data_financial_news.columns.unique()))


# In[36]:


# Generator
generator = FullBatchNodeGenerator(financial_news_graph, method = "gcn")


# # Machine Learning, Deep Learning, GCN, CNN

# # Train Test Split

# In[37]:


train_subjects, test_subjects = model_selection.train_test_split(
    node_Data_financial_news #, train_size = 6, test_size = 4
)
# , train_size=6, test_size=None, stratify=pearson_graph_node_data

val_subjects, test_subjects_step_2 = model_selection.train_test_split(
    test_subjects #, test_size = 2
)

#, train_size = 500, test_size = None, stratify = test_subjects


train_subjects.shape, test_subjects.shape, val_subjects.shape, test_subjects_step_2.shape


# In[38]:


# just the target variables

train_targets = train_subjects; 
val_targets = val_subjects; 
test_targets = test_subjects; 


# In[39]:


# Architecture of the Neural Network
train_subjects.index, train_targets


# In[40]:


train_gen = generator.flow(train_subjects.index, train_targets)


# In[41]:


from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, Dropout, Flatten
from tensorflow import keras

layer_sizes = [32, 32]
activations = ["relu", "relu"]

gcn = GCN(layer_sizes = layer_sizes, activations = activations, generator = generator) #, dropout = 0.5
x_inp, x_out = gcn.in_out_tensors()

# MLP -- Regression
predictions = layers.Dense(units = train_targets.shape[1], activation = "linear")(x_out)
x_out, 
x_inp, x_out


# # Models

# In[42]:


# loss functions: https://keras.io/api/losses/

model = Model(
    inputs = x_inp, outputs = predictions)

'''
model.compile(
    optimizer=optimizers.Adam(learning_rate=0.1),
    loss=losses.MeanSquaredError(),
    metrics=["acc"],
)
'''

# REF: https://stackoverflow.com/questions/57301698/how-to-change-a-learning-rate-for-adam-in-tf2
# https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/PolynomialDecay
train_steps = 1000
lr_fn = optimizers.schedules.PolynomialDecay(1e-3, train_steps, 1e-5, 2)


# https://keras.io/api/metrics/
model.compile(
    loss = 'mean_absolute_error', 
    optimizer = optimizers.Adam( lr_fn ),
    # metrics = ['mean_squared_error']
    metrics=['mse', 'mae', 'mape']
)

# mape: https://towardsdatascience.com/choosing-the-correct-error-metric-mape-vs-smape-5328dec53fac
model.compile( 
    loss = 'mean_absolute_error', 
    optimizer = optimizers.Adam(learning_rate = 0.015), 
    #optimizer = optimizers.Adam(lr_fn), 
    # metrics=['mean_squared_error']
    metrics=['mean_squared_error', 'mae', 'mape']
    # metrics=[
    #    metrics.MeanSquaredError(),
    #    metrics.AUC(),
    #]
)


# In[43]:


len(x_inp), predictions.shape, print(model.summary())


# In[44]:


len(val_subjects)
test_subjects_ = test_subjects[:len(val_subjects)]


# In[45]:


# hard coded size adjustments
test_subjects_ = test_subjects[:len(val_subjects)]

val_gen = generator.flow(val_subjects.index, test_subjects_)
#train_gen[1], val_gen[1]


# In[46]:


epochs_to_test = 1000
patience_to_test = 1000
data_valid = val_gen #[:1][:4];
train_gen_data = train_gen #[:1][:4];


# In[47]:


type(train_gen_data), type(data_valid), type(x_inp), type(x_out) 


# In[48]:


# https://keras.io/api/callbacks/early_stopping/
from tensorflow.keras.callbacks import EarlyStopping

es_callback = EarlyStopping(
    monitor = "val_mean_squared_error", 
    patience = patience_to_test, 
    restore_best_weights = True
)

history = model.fit( train_gen_data, epochs = epochs_to_test, validation_data = data_valid, verbose = 2,    
    # shuffling = true means shuffling the whole graph
    shuffle = False, callbacks = [es_callback],
)
sg.utils.plot_history(history)

# [1]


# In[49]:


val_subjects, 
test_subjects


# In[50]:


test_gen = generator.flow(test_subjects.index, test_targets)
test_metrics = model.evaluate(test_gen)
print("\nTest Set Metrics:")
for name, val in zip(model.metrics_names, test_metrics):
    print("\t{}: {:0.4f}".format(name, val))
    


# # Show the predicted prices by the Model
# 
# At this point, I still need to make sense of what GCN ( and CNN) combination + MLP is predicting. 
# I am just displaying the output. 
# It appears that price is predicted for each timestamp (day)

# In[51]:


all_nodes = node_Data_financial_news.index;
all_gen = generator.flow(all_nodes)
all_predictions = model.predict(all_gen)

all_predictions, all_predictions.shape, node_Data_financial_news.shape


# In[52]:


df_metrics = pd.DataFrame(columns=['Method', 'Loss', 'MSE', 'MAE', 'MAPE'])

temp = list()
temp.append('GCN-Causation-News');
for name, val in zip(model.metrics_names, test_metrics):    
    temp.append(val)

print(temp)
df_metrics.loc[1] = temp


# In[ ]:





# In[53]:


import math
df_metrics_plot = df_metrics[['Loss', 'MSE', 'MAE', 'MAPE']]
df_metrics_plot['MSE'] = math.sqrt(df_metrics['MSE'])
df_metrics_plot


# In[54]:


df_metrics_plot.plot( kind = 'bar')


# In[ ]:




