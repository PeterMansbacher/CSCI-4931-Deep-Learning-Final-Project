# CSCI-4931-Deep-Learning-Final-Project
*******************************************************
*  Name      : Peter Mansbacher  & Manthan Baboo       
*  Student ID: 104405435                
*  Class     :  CSCI 4931           
*  HW#       :  Final Project         
*  Due Date  :  May 13, 2021
*******************************************************


                 Read Me


*******************************************************
*  Description of the program
*******************************************************
Our problem was, how can we use already aired episodes of a TV show to create ideas for new, non-canon or canon
episodes in the series. We solved this by using around 500 episode descriptions of Naruto Shippuden 
from IMDb as data for training Vanilla Recurrent Neural Networks. The results should produce new descriptions
for potential episodes, although the Vanilla architecture doesn't create as sensibly structured sentences 
as other RNN architectures. Citations included.




*******************************************************
*  Source files
*******************************************************

Name: Project_Mansbacher_Baboo.ipynb
  This Jupyter Notebook contains the project source code
  
Name:Naruto_Episodes_IMDB_Descriptions.csv
  This contains the scrapped descriptions only (still contains stopwords)

Name:Naruto_Episodes_IMDB.csv
  This csv file contains the scrapped season,	episode_number,	title, and	description.



   
*******************************************************
*  Circumstances of programs
*******************************************************

   The program runs successfully, but may require a long time to train model.
   
   The program was developed and tested with Jupyter. 
   
   
 

*******************************************************
*  How to build and run the program
*******************************************************

Program requires certain libraries listed in source code and below.
Import these like in source code, not like below; these are for reference. 


    "import numpy as np\n",
    "import tensorflow as tf\n",
    "import pandas as pd\n",
    "from tensorflow import keras\n",
    "from tensorflow.keras import layers\n",
    "from keras.models import Sequential\n",
    "from keras.layers import Dense, Dropout, LSTM\n",
    "from keras.utils import np_utils\n",
    "from keras.callbacks import ModelCheckpoint\n",
    "\n",
    "\n",
    "#scrapping tools etc from isabella-b.com link\n",
    "from requests import get\n",
    "from bs4 import BeautifulSoup\n",
    "import requests\n",
    "import nltk\n",
    "from nltk.tokenize import RegexpTokenizer\n",
    "\n",
    "from sklearn.metrics.pairwise import cosine_similarity\n",
    "from nltk.stem import PorterStemmer\n",
    "from nltk.corpus import stopwords\n",
    "nltk.download('stopwords')\n",
    "nltk.download('punkt')\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "\n",
    "#lec 17\n",
    "import csv\n",
    "import itertools\n",
    "import operator\n",
    "import sys\n",
    "from datetime import datetime\n",
    "\n",
    "import matplotlib.pyplot as plt"

	
