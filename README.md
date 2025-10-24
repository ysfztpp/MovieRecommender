# Cinematch AI: Real-Time Content-Based Movie Recommender

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://streamlit.io)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-blue?logo=meta)](https://faiss.ai/)
[![Pandas](https://img.shields.io/badge/Pandas-Data_Processing-blue?logo=pandas)](https://pandas.pydata.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Preprocessing-orange?logo=scikit-learn)](https://scikit-learn.org/)

A high-performance, content-based movie recommendation engine built with **FAISS (Meta AI)** to deliver instant search results from the **MovieLens 20M dataset**.

Most recommenders fail when faced with real-world scale. A standard K-Nearest Neighbors (K-NN) search on 27,000+ films is computationally slow and cannot provide real-time results.

This project's core is a vector search engine that pre-calculates and indexes all 27,000 film vectors, enabling **instant similarity search** using Facebook AI's (Meta) FAISS library.

***

###  Demo
https://ibb.co/mF2mFqCm

---

## 🚀 The Core Challenge: Why FAISS?

The goal was to build a *content-based* recommender using the rich **1129-dimension "genome" tags** from the 27,000-film dataset.

* **The Problem:** A standard `sklearn.KNeighbors` (K-NN) model must perform a brute-force search, calculating the distance between the query film and all 27,000 other films *every single time* a user searches. This is far too slow for a real-time web app.
* **The Solution:** FAISS allows us to build a high-dimensional index (in this case, `IndexFlatL2`) from all 27,000 film vectors *once*. At runtime, the Streamlit app only needs to query this pre-built index, which returns the 'k' nearest neighbors almost instantly.

This project is a practical demonstration of moving from a slow, academic model (K-NN) to a high-performance, production-ready system (FAISS).

---

## 🛠 Tech Stack

* **Vector Search Engine:** `faiss-cpu` (Meta AI)
* **Data Processing:** `pandas`, `numpy`
* **Feature Engineering:** `scikit-learn` (for preprocessing and L2 normalization)
* **Web Interface:** `streamlit`
* **Data Source:** [MovieLens 20M Dataset](https://grouplens.org/datasets/movielens/20m/), [TMDB API](https://www.themoviedb.org/documentation/api) (for posters)

---

## ⚙️ How It Works: Architecture

1.  **Data Preprocessing:**
    * Film metadata (genres, year) and genome tags (1129 dimensions) are loaded from the MovieLens dataset.
    * **Data Imputation:** Missing genre data is imputed by analyzing the preferences of users who frequently rated those films.
    * Genres are One-Hot Encoded.

2.  **Vectorization:**
    * All features (genres, year, and 1129-dim tags) are combined into a single, high-dimensional vector for each of the 27,000 films.
    * Vectors are normalized using **L2 Normalization** (via `scikit-learn`) to ensure the FAISS index calculates similarity based on Euclidean distance correctly.

3.  **Indexing (The Core):**
    * All 27,000 normalized vectors are loaded into a FAISS `IndexFlatL2` object.
    * This index is saved to disk (`index.faiss`) so the app can load it instantly without re-training.

4.  **Search (Runtime):**
    * The Streamlit app loads the pre-built FAISS index.
    * When a user selects a film, its vector is retrieved and used to query the index.
    * `index.search()` instantly returns the IDs of the 10 most similar films, which are then used to pull metadata and posters from the TMDB API.

---

## 🚀 How to Run Locally

You will need Python 3.9+, the MovieLens 20M dataset, and a TMDB API key.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/cinematch-ai.git](https://github.com/your-username/cinematch-ai.git)
    cd cinematch-ai
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(This will install `streamlit`, `pandas`, `numpy`, `scikit-learn`, and `faiss-cpu`)*

3.  **Set up your data:**
    * Download the [MovieLens 20M dataset](https://files.grouplens.org/datasets/movielens/ml-20m.zip) and extract it to a `data/` folder in the project directory.
    * You will need to run the `preprocessing.py` script first to generate the vectors and the FAISS index file.
    ```bash
    python preprocessing.py
    ```
    *(This only needs to be run once. It will create `index.faiss` and other processed data files.)*

4.  **Set up your API Key:**
    * Get a free API key from [TMDB](https://www.themoviedb.org/documentation/api).
    * Create a file named `.env` in the root directory and add your key:
    ```
    TMDB_API_KEY="your_key_here"
    ```

5.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```
    Open your browser to `http://localhost:8501` and enjoy!

---

## ⚖️ License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.