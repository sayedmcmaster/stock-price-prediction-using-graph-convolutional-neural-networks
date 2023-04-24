By Sayed Ahmed (400290668)

# stock-price-prediction-using-graph-convolutional-neural-networks

1. Steps to execute the project

1.1 Make sure the data folder has the corresponding data files mentioned in the code

1.2. Then using Jupyter Notebook execute the code.

1.3. It is important to see if the required software and modules are there or not. Check the software requirements section

Note:
The folder: execution-pdfs will have some pdfs created with the ipynb execution outputs. These can also be found at:

https://github.com/sayedmcmaster/stock-price-prediction-using-graph-convolutional-neural-networks/tree/main/submission/execution-pdfs


 
**4. Software/Tools Requirements**

Anaconda Navigator (optional, I used this)

Jupyter Notebook or a tool that can execute ipynb files


Python <= 3.8, latest versions may not work as some libraries need earlier versions

Steller Graph, the latest version can be used

Tensorflow

Keras

Torch/Pytorch optionally may be needed

CV2 or similar

yfinance module


Data Files:
[Dataset adjustments, and aligning data to the project and model took a significant amount of time and efforts (to start with)]
data/per-day-fortune-30-company-stock-price-data.csv

data/yahoonewsarchive/News_Yahoo_stock.csv

------
Previously used data files i.e. dataset
./data/stock-price--all-merged.csv (still may work although need more adjustments for good/correct results)
./data/small-stock-price--all-merged.csv

The folder: C:\my_works_2021_07_25\hd-thing\mcmaster\meng\747\project\submission\archive\stock_market_data\nasdaq\csv has individual stock data that was used to create stock-price--all-merged.csv.

Althugh these data can be used and may work or may be made to work wit minor adjustments, I no longer used these data in the final stage.
-----


**2. Important Code Files**

**simplest_model_gcn_mlp_only-stock-prediction-cleaned.ipynb**
-- It used a simple GCN and MLP layer to predict prices. It works on Pearson, Spearman, and Kendall tau-based stock price prediction. Also, plots and compares the performance. I also have integrated financial news-based prediction into this same file to compare all in one file.

On another note, to start with code files for Pearson, Spearman, Kendal, and News were separate; later merged into one. The code File is now large; although, I am still doing so that I can plot the performance together on the same file.

2nd-model-stock-prediction-cleaned.ipynb :  A different model to predict stock prices. This model has GCN + CNN + MLP layers. This is my closest model/work reflecting the paper. The paper uses multiple blocks of GCN + CNN and at the end one MLP. I use GCN + CNN + MLP layers. My GCN uses multiple layers such as 32 layers.

final-more-cleaned-3rd-model-deep-graph-cnn-stock-prediction-cleaned.ipynb  (one step before: more-cleaned-3rd-model-deep-graph-cnn-stock-prediction-cleaned.ipynb) : A different model named DeepGraphCNN to predict stock prices. DeepGraphCNN is a built-in model in StellarGraph Library. I have customized it for my purpose. This approach needs much more work for correct prediction although at the end I could progress well. It uses multiple train, test, and validation graphs. I have only one graph that I divided into one train, one test, and one validation graph. However, more graphs either by dividing/chunking the time series or using other prices such as adjusted close prie, open price, or even per-minute prices could lead to more graphs that could effectively (likely) predict stock prices closely. 

**fortune-30-company-stock-data.ipynb**
-- I wrote this code to scrap stock price data from Yahoo. I did collect per-day interval data as well as per-minute interval data. I collected data for the same stocks the paper used. However, data was not found for two stocks.


**misc-early-step-code-got-merged-into-new-code/yfinance_find-stock-tickers-in-news-articles-create-graph-predict-with-gcn.ipynb**
-- This file involves stock price prediction using Financial News data. I have integrated this code into other files that predicts using news data such as  **simplest_model_gcn_mlp_only-stock-prediction-cleaned.ipynb** and 2nd-model-stock-prediction-cleaned.ipynb


**merge-only-matching-csv-files-with-python.ipynb**
-- This file is no longer used; however, it was used for a good amount of time. From the downloaded dataset (from Kaggle/yahoo or so), I used this code to merge multiple individual stock data into one file and create the dataset for the project. It can take a list of stocks to merge as input and can combine them into a dataset. 

**merge-multiple-csv-files-with-python.ipynb**
-- like the previous one this also merges multiple stock price data in individual files into a combined dataset. It takes all files in the selected folder. It does not take an input list of stocks to work with.

**find-stock-tickers-in-news-articles.ipynb**
--- I wrote this code to find stock tickers in financial news articles. Then creates weighted graphs based on how those stocks are related/mentioned together.
Later I took this code to integrate into **yfinance_find-stock-tickers-in-news-articles-create-graph-predict-with-gcn.ipynb** which will eventually be integrated into 
**simplest_model_gcn_mlp_only-stock-prediction-cleaned.ipynb**


Note: My Explanation (video) of the code can be seen on: https://www.youtube.com/playlist?list=PLUA7SYgJYDFoAmJSyCXtKwA1uw2ErFgEo

Most steps, most codes, and most results are explained there.


**3. Data Files**
-- Data files are kept in the data folder. The code usually loaded data in the beginning of the code.


Github folder: https://github.com/sayedmcmaster/stock-price-prediction-using-graph-convolutional-neural-networks/tree/main/submission will have the submitted version of the code and data.

Github location: https://github.com/sayedmcmaster/stock-price-prediction-using-graph-convolutional-neural-networks may have more code and data.

Execution and Outputs are kept as PDFs at:
https://github.com/sayedmcmaster/stock-price-prediction-using-graph-convolutional-neural-networks/tree/main/submission/execution-pdfs



