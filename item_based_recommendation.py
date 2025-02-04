import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from scipy.sparse import hstack
from scipy.sparse import csr_matrix

# Load the data
data = pd.read_csv('Book_recom.csv')

# Handle missing values
data.fillna({'Book-Author': '', 'Publisher': '', 'Location': '', 'Popularity': 0, 'Age': data['Age'].mean()}, inplace=True)

# Prepare categorical, numerical, and text data
categorical_features = data[['Book-Author', 'Publisher', 'Location']]
numerical_features = data[['Year-Of-Publication', 'Popularity', 'Age', 'Title-Sentiment']]
text_features = data['Book-Title']  # Use book titles for TF-IDF

# One-hot encode categorical features
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
encoded_categorical = encoder.fit_transform(categorical_features)

# Normalize numerical features
scaler = MinMaxScaler()
scaled_numerical = scaler.fit_transform(numerical_features)

# Apply TF-IDF to text features
tfidf = TfidfVectorizer(max_features=100)
tfidf_text = tfidf.fit_transform(text_features)

# Combine all features
combined_features = hstack([encoded_categorical, scaled_numerical, tfidf_text])

# Fit kNN model
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=3)
model_knn.fit(combined_features)

# Convert combined_features to CSR format after stacking
combined_features = csr_matrix(combined_features)

# Update the find_similar_books function
def find_similar_books(target_isbn, n_neighbors=7):
    if target_isbn in data['ISBN'].values:
        target_idx = data.index[data['ISBN'] == target_isbn][0]
        
        # Use CSR slicing to get a single row as a 2D matrix
        target_vector = combined_features[target_idx]

        distances, indices = model_knn.kneighbors(target_vector, n_neighbors=n_neighbors)

        print(f"Target Book: {data.loc[target_idx, 'Book-Title']} by {data.loc[target_idx, 'Book-Author']}")
        print("\nSimilar Books:")
        for i, index in enumerate(indices[0][1:], start=1):  # Skip the target book itself
            similar_book = data.iloc[index]
            print(f"{i}: {similar_book['Book-Title']} by {similar_book['Book-Author']} (Distance: {distances[0][i]:.2f})")
    else:
        print(f"Book with ISBN {target_isbn} not found in the dataset.")

# Update the recommend_books_for_user function similarly
def recommend_books_for_user(user_id, n_neighbors=5):
    if user_id in data['User-ID'].values:
        # print(f"User {user_id} found in dataset")
        user_books = data[data['User-ID'] == user_id]
        rated_books = user_books['ISBN'].values
        # print(f"User {user_id} rated books: {rated_books}")

        recommendations = {}
        for book in rated_books:
            if book in data['ISBN'].values:
                # print(f"Processing book {book}")
                target_idx = data.index[data['ISBN'] == book][0]

                # Use CSR slicing to get a single row as a 2D matrix
                target_vector = combined_features[target_idx:target_idx + 1]

                distances, indices = model_knn.kneighbors(target_vector, n_neighbors=n_neighbors)
                # print(f"Distances: {distances}")
                # print(f"Indices: {indices}")

                for i, index in enumerate(indices[0][1:], start=1):  # Skip the target book itself
                    similar_book = data.iloc[index]
                    if similar_book['ISBN'] not in rated_books:  # Avoid recommending already rated books
                        recommendations[similar_book['ISBN']] = recommendations.get(similar_book['ISBN'], 0) + 1 / distances[0][i]

        # Sort recommendations by score
        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        if sorted_recommendations:
            print("\nRecommended Books:")
            for isbn, score in sorted_recommendations[:n_neighbors]:
                book_info = data[data['ISBN'] == isbn].iloc[0]
                print(f"ISBN: {isbn}, Title: {book_info['Book-Title']} by {book_info['Book-Author']}")
        else:
            print("No recommendations could be generated.")
    else:
        print(f"User with ID {user_id} not found in the dataset.")

target_isbn = "2080674722"  # Replace with an actual ISBN from your dataset
find_similar_books(target_isbn)

user_id = 276762  # Replace with an actual User-ID from your dataset
recommend_books_for_user(user_id)