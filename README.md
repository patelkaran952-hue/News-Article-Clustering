# News Article Clustering using NLP and Unsupervised Machine Learning

## Project Overview

This project performs unsupervised clustering of news articles from the BBC News Dataset using Natural Language Processing (NLP) techniques and machine learning algorithms.

The objective is to automatically group similar news articles without using category labels during training.

The clustering process successfully discovers hidden topics such as:

* Business
* Entertainment
* Politics
* Sport
* Technology

---

## Dataset

**BBC News Dataset**

* Total Articles: 2,225
* Categories: 5
* Data Type: Text Documents

| Category      | Articles |
| ------------- | -------- |
| Business      | 510      |
| Entertainment | 386      |
| Politics      | 417      |
| Sport         | 511      |
| Technology    | 401      |

---

## Problem Statement

Given a collection of unlabeled news articles, the objective is to automatically discover groups of similar articles using unsupervised learning techniques.

---

## Technologies Used

* Python
* Pandas
* NumPy
* NLTK
* Scikit-Learn
* Matplotlib
* Seaborn

---

## NLP Pipeline

### 1. Data Loading

* Read articles from text files
* Create a Pandas DataFrame

### 2. Data Cleaning

* Convert text to lowercase
* Remove punctuation
* Remove numbers
* Remove stopwords
* Lemmatization

### 3. Feature Extraction

* TF-IDF Vectorization
* Unigrams and Bigrams
* Vocabulary size control

### 4. Dimensionality Reduction

* Truncated SVD (Latent Semantic Analysis)

### 5. Clustering Algorithms

* K-Means Clustering
* Agglomerative Hierarchical Clustering

### 6. Evaluation

* Silhouette Score

### 7. Visualization

* PCA Visualization
* Cluster Scatter Plots
* Dendrogram
* Top Keywords per Cluster

---

## Project Structure

```text
News-Article-Clustering/
│
├── data/
│   └── bbc/
│
├── notebooks/
│   └── News_Article_Clustering.ipynb
│
│
├── requirements.txt
├── README.md
└── main.py
```

---

## Data Preprocessing

```python
def clean_text(text):

    text = text.lower()

    text = re.sub(r'[^a-zA-Z]', ' ', text)

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)
```

---

## TF-IDF Vectorization

```python
vectorizer = TfidfVectorizer(
    max_features=5000,
    min_df=3,
    max_df=0.8,
    ngram_range=(1,2)
)

X = vectorizer.fit_transform(
    bbc['clean_text']
)
```

---

## Dimensionality Reduction

```python
svd = TruncatedSVD(
    n_components=100,
    random_state=42
)

X_reduced = svd.fit_transform(X)
```

---

## K-Means Clustering

```python
kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=20
)

bbc['KMeans_Cluster'] = kmeans.fit_predict(
    X_reduced
)
```

---

## Results

* Successfully clustered news articles into meaningful groups.
* Reduced dimensionality using Latent Semantic Analysis.
* Extracted top keywords from each cluster for interpretation.

Example cluster interpretation:

| Cluster   | Topic         |
| --------- | ------------- |
| Cluster 0 | Sport         |
| Cluster 1 | Politics      |
| Cluster 2 | Business      |
| Cluster 3 | Entertainment |
| Cluster 4 | Technology    |

---

## Example Keywords Discovered

### Cluster 0

```
game
player
match
club
team
cup
```

### Cluster 1

```
government
election
party
minister
labour
```

### Cluster 2

```
market
company
profit
share
bank
```

### Cluster 3

```
film
music
award
actor
show
```

### Cluster 4

```
software
computer
internet
mobile
technology
```

---

## Future Improvements

* BERTopic
* Sentence Transformers Embeddings
* UMAP Visualization
* HDBSCAN Clustering
* Interactive Dashboard using Streamlit

---

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python main.py
```

---

## Learning Outcomes

This project demonstrates understanding of:

* Natural Language Processing
* Text Preprocessing
* TF-IDF Vectorization
* Dimensionality Reduction
* Unsupervised Machine Learning
* Document Clustering
* Cluster Evaluation
* Data Visualization

---

## Author

Karan Patel

If you found this project useful, consider giving it a star.
