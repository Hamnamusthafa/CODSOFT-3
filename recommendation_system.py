# recommendation_system.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Sample movie dataset (you can expand this later)
data = {
    "movie_id": [1, 2, 3, 4, 5, 6],
    "title": [
        "Avengers",
        "Iron Man",
        "Captain America",
        "Thor",
        "Spider-Man",
        "Doctor Strange"
    ],
    "genres": [
        "Action Sci-Fi",
        "Action Sci-Fi",
        "Action Adventure",
        "Action Fantasy",
        "Action Adventure",
        "Fantasy Sci-Fi"
    ]
}

df = pd.DataFrame(data)

# Step 2: Convert genres text to numerical features using TF-IDF
vectorizer = TfidfVectorizer()
genre_matrix = vectorizer.fit_transform(df["genres"])

# Step 3: Calculate similarity between all movies
similarity = cosine_similarity(genre_matrix)

# Step 4: Recommendation function
def recommend(movie_title):
    if movie_title not in df["title"].values:
        print("❌ Movie not found in our database.")
        return

    # Get index of selected movie
    idx = df[df["title"] == movie_title].index[0]

    # Get similarity scores for this movie
    sim_scores = list(enumerate(similarity[idx]))

    # Sort by highest similarity score (excluding the movie itself)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    print(f"\n🎬 Recommended movies for '{movie_title}':")
    count = 0
    for i in sim_scores[1:]:
        recommended_title = df.iloc[i[0]]["title"]
        print("-", recommended_title)
        count += 1
        if count == 3:
            break

# Step 5: Run the program
if __name__ == "__main__":
    print("📽️ Welcome to the Movie Recommendation System!\n")
    print("🎞️ Available movies:")
    for title in df["title"]:
        print("-", title)

    # Ask user input
    user_input = input("\n🤔 Enter a movie you like: ").strip()
    recommend(user_input)
