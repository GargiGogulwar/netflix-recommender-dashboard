#  Netflix Content Recommender + Insights Dashboard



An interactive Streamlit dashboard to explore Netflix titles, visualize trends,

and get content-based recommendations using genres and descriptions.



## Dataset



Uses the "Netflix TV Shows and Movies" dataset (Victor Soeiro) from Kaggle.  

Download `titles.csv` from Kaggle and place it in the `data/` folder:



`data/titles.csv`



## Features



\- 📊 EDA \& Insights

&nbsp; - Titles per release year

&nbsp; - Top genres

&nbsp; - IMDB score distribution

\- 🎯 Content-based recommender

&nbsp; - TF-IDF on genres + descriptions

&nbsp; - Cosine similarity to find similar titles

\- 🧩 Clustering

&nbsp; - KMeans clustering on year, runtime, IMDB \& TMDB scores



## How to Run



```bash

# clone the repo

git clone <your-repo-url>

cd netflix-recommender-dashboard



# create and activate virtualenv

python -m venv venv

.\venv\Scripts\Activate.ps1  # PowerShell

# or

venv\Scripts\activate.bat    # CMD



# install dependencies

pip install -r requirements.txt



# add dataset

# place titles.csv into data/titles.csv



# run the app

streamlit run app\streamlit_app.py



