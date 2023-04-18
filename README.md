By Sayed Ahmed

# stock-price-prediction-using-graph-convolutional-neural-networks

1. Steps to execute the project

1.1 Make sure the data folder has the the corresponding data files mentioned in the code

1.2. Then using Jupyter Notebook execute the code.

1.3. It is important to see if the required software and modules are there or not. Check the software requirements section
 

**2. Important Code Files**

**simplest_model_gcn_mlp_only-stock-prediction-cleaned.ipynb**
-- It used a simple GCN and MLP layer to predict prices. It works on Pearson, Spearman, and kendall tau based stock price prediction. Also, plots and compares the performance. I will integrating financial news based prediction into this same file to compare all in one file.

On another note, for each model i.e.style of prediction I will create one file and put all Types of Graph based prediction code into the same file. Code File will be large; although, I may still do so that I can plot the performance on the same file.

2nd-model-stock-prediction-cleaned.ipynb :  A different model to predict stock prices. This model has GCN + CNN + MLP layers. This is my closest work reflecting the paper. The paper uses multiple blocks of GCN + CNN and at the end one MLP. My GCN uses multiple layers such as 32 layers.

3rd-model-deep-graph-cnn-stock-prediction-cleaned.ipynb : A different model named DeepGraphCNN to predict stock prices. DeepGraphCNN is a built-in model in StellarGraph Library. I have customized it for my purpose.



**yfinance_find-stock-tickers-in-news-articles-create-graph-predict-with-gcn.ipynb**
-- This file involves stock price prediction using Financial News data. It used to work using a different/older dataset/approach. However, the prediction was not correct. I am using a different approach and/or a newer dataset to make it work. If it works, I will integrate this code into the file **simplest_model_gcn_mlp_only-stock-prediction-cleaned.ipynb**

**fortune-30-company-stock-data.ipynb**
-- I wrote this code to scrap stock price data from yahoo. I did collect per day interval data as well as per minute interval data. I collected data for the same stocks the paper used. However, data was not found for two stocks

**merge-only-matching-csv-files-with-python.ipynb**
-- This file is no longer used; However, it was used for a good amount of time. From downloaded dataset (from Kaggle or so), I used this code to merge multiple individual stock data into one file and create the dataset for the project. It can take a list of stocks to merge as input and can combine them into a dataset. 

**merge-multiple-csv-files-with-python.ipynb**
-- like the previous one this also merges multiple stock price data in individual files to a combined dataset. It takes all files in the selected folder. It does not take an input list of stocks to work with.

**find-stock-tickers-in-news-articles.ipynb**
--- I wrote this code to find stock tickers in financial news articles. Then creates weighted graphs based on how those stcoks are related/mentioned together.
Later I took this code to integrate into **yfinance_find-stock-tickers-in-news-articles-create-graph-predict-with-gcn.ipynb** which will eventually be integrated into 
**simplest_model_gcn_mlp_only-stock-prediction-cleaned.ipynb**


Note: My Explanation of the code can be see on: https://www.youtube.com/playlist?list=PLUA7SYgJYDFoAmJSyCXtKwA1uw2ErFgEo

**3. Data Files**
-- Data files are kept in the data folder. The code usually loaded data in the beginning of the code.


**4. Software/Tools Requirements**

Anaconda Navigator (optional, I used this)

Jupyter Notebook or a tool that can execute ipynb files


Python <= 3.8, latest versions may not work as some libraries need earlier versions

Steller Graph, latest version can be used

Tensorflow

Keras

Torch/Pytorch optionally may be needed

CV2 or similar

yfinance module
