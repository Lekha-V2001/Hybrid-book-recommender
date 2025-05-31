import streamlit as st
import pandas as pd
from surprise import Dataset, Reader, KNNBasic
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page Setup
st.set_page_config(page_title="📚 Book Recommender", layout="wide")
st.markdown(
    "<h1 style='text-align: center; color: #4B8BBE;'>📚 Hybrid Book Recommender</h1><hr>",
    unsafe_allow_html=True,
)

# Load data with error handling
try:
    books = pd.read_csv("Books.csv", nrows=10000, encoding="latin-1")
    ratings = pd.read_csv("Ratings.csv", nrows=10000, encoding="latin-1")
    ratings = ratings[ratings['Book-Rating'] > 0]
    users = pd.read_csv("Users.csv", nrows=10000, encoding="latin-1")
except FileNotFoundError as e:
    st.error(f"Data file not found: {e.filename}")
    st.stop()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Collaborative Filtering Setup
reader = Reader(rating_scale=(0, 10))
data = Dataset.load_from_df(ratings[['User-ID', 'ISBN', 'Book-Rating']], reader)
trainset = data.build_full_trainset()
cf_model = KNNBasic(sim_options={'user_based': False})
cf_model.fit(trainset)

# Content-Based Filtering Setup
books['Book-Title'] = books['Book-Title'].astype(str)
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(books['Book-Title'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
book_indices = pd.Series(books.index, index=books['Book-Title']).drop_duplicates()

# Hybrid Recommendation Function
def hybrid_recommender(user_id, book_title, top_n=20):
    idx = book_indices.get(book_title)
    if idx is None:
        return []
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    candidate_isbns = [books.iloc[i[0]]['ISBN'] for i in sim_scores]
    recs = []
    for isbn in candidate_isbns:
        try:
            pred = cf_model.predict(user_id, isbn)
            recs.append((isbn, pred.est))
        except:
            continue
    return sorted(recs, key=lambda x: x[1], reverse=True)

# Sidebar: Filters and Top Books
with st.sidebar:
    st.markdown("## 🔧 Customize Your Preferences")
    user_id = st.number_input("👤 Enter Your User ID", 
                              min_value=int(ratings['User-ID'].min()), 
                              max_value=int(ratings['User-ID'].max()),
                              value=int(ratings['User-ID'].min()))
    
    book_title = st.selectbox("📖 Pick a Book You Like", books['Book-Title'].unique())

    min_rating = st.slider("⭐ Minimum Predicted Rating", 1.0, 10.0, 7.0, 0.5)
    num_rec = st.number_input("🔢 Number of Recommendations", 1, 20, 5)
    
    st.markdown("---")
    st.markdown("## 📈 Most Rated Books")
    N = st.slider("Top N Books", 5, 20, 10)
    top_counts = ratings['ISBN'].value_counts().head(N)
    available_isbns = top_counts.index.intersection(books['ISBN'])
    top_books = books.set_index('ISBN').loc[available_isbns]['Book-Title']
    for i, title in enumerate(top_books, 1):
        st.markdown(f"{i}. **{title}**")

# Rating Distribution Chart
st.markdown("### 📊 Overall Book Rating Distribution")
rating_counts = ratings['Book-Rating'].value_counts().sort_index()
st.bar_chart(rating_counts)

# Recommendations Section
st.markdown("---")
if st.sidebar.button("🎯 Get Recommendations"):
    raw_recs = hybrid_recommender(user_id, book_title, top_n=20)
    filtered = [(isbn, score) for isbn, score in raw_recs if score >= min_rating][:num_rec]
    
    if not filtered:
        st.warning(f"No books found with predicted rating ≥ {min_rating}")
    else:
        st.markdown("## 🔖 Recommended Books For You")
        cols = st.columns(2)
        for idx, (isbn, score) in enumerate(filtered, 1):
            match = books.loc[books['ISBN'] == isbn, 'Book-Title']
            title = match.values[0] if not match.empty else "Unknown Title"
            with cols[idx % 2]:
                st.markdown(f"""
                <div style='background-color:#f9f9f9;padding:10px 15px;margin:10px 0;border-radius:10px;border-left:5px solid #4B8BBE;'>
                    <h4>{idx}. {title}</h4>
                    <p><strong>Predicted Rating:</strong> {score:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
