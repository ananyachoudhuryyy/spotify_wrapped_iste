import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import ast
import warnings
warnings.filterwarnings('ignore')

class SpotifyMusicRecommender:
    def __init__(self, data_path):
        """Initialize the recommender system with the dataset."""
        print("Loading dataset...")
        self.df = pd.read_csv(data_path)
        print(f"Loaded {len(self.df)} songs")
        
        # Audio features to use for similarity
        self.feature_cols = ['valence', 'acousticness', 'danceability', 'energy', 
                            'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo']
        
        # Prepare the data
        self.prepare_data()
        
    def prepare_data(self):
        """Prepare and scale the feature data."""
        print("Preparing features...")
        
        # Extract features
        self.features = self.df[self.feature_cols].copy()
        
        # Handle any missing values
        self.features = self.features.fillna(self.features.mean())
        
        # Scale features
        self.scaler = StandardScaler()
        self.scaled_features = self.scaler.fit_transform(self.features)
        
        # Build KNN model
        print("Building KNN model...")
        self.knn = NearestNeighbors(n_neighbors=50, metric='euclidean')
        self.knn.fit(self.scaled_features)
        
    def get_random_songs(self, n=10):
        """Get n random songs for rating."""
        random_indices = np.random.choice(len(self.df), size=n, replace=False)
        return self.df.iloc[random_indices].copy()
    
    def display_song(self, song, index=None):
        """Display song information in a readable format."""
        if index is not None:
            print(f"\n{index}.")
        
        # Parse artists if it's a string representation of a list
        artists = song['artists']
        if isinstance(artists, str):
            try:
                artists = ast.literal_eval(artists)
                if isinstance(artists, list):
                    artists = ', '.join(artists)
            except:
                pass
        
        print(f"  Song: {song['name']}")
        print(f"  Artist(s): {artists}")
        print(f"  Year: {int(song['year'])}")
        print(f"  Popularity: {song['popularity']}")
        
    def get_user_ratings(self):
        """Get user ratings for 20 random songs."""
        print("\n" + "="*70)
        print("MUSIC RECOMMENDATION SYSTEM")
        print("="*70)
        print("\nYou will be shown 10 random songs.")
        print("Please rate each song on a scale of 1-5:")
        print("  1 = Don't like at all")
        print("  2 = Don't really like")
        print("  3 = It's okay")
        print("  4 = Like it")
        print("  5 = Love it")
        print("\nIf you haven't heard the song, give your best guess based on the artist/info!")
        print("="*70)
        
        random_songs = self.get_random_songs(20)
        ratings = []
        rated_songs = []
        
        for idx, (_, song) in enumerate(random_songs.iterrows(), 1):
            self.display_song(song, idx)
            
            while True:
                try:
                    rating = input("  Your rating (1-5): ").strip()
                    rating = int(rating)
                    if 1 <= rating <= 5:
                        ratings.append(rating)
                        rated_songs.append(song)
                        break
                    else:
                        print("  Please enter a number between 1 and 5.")
                except ValueError:
                    print("  Please enter a valid number.")
        
        return pd.DataFrame(rated_songs), np.array(ratings)
    
    def recommend_songs(self, rated_songs_df, ratings, n_recommendations=10):
        """Recommend songs based on user ratings using weighted KNN."""
        print("\n" + "="*70)
        print("GENERATING RECOMMENDATIONS...")
        print("="*70)
        
        # Get indices of rated songs
        rated_indices = rated_songs_df.index.tolist()
        
        # Normalize ratings to use as weights (higher rating = higher weight)
        weights = ratings / ratings.sum()
        
        # Get scaled features for rated songs
        rated_features = self.scaled_features[rated_indices]
        
        # Calculate weighted average of liked songs' features
        # Give more weight to higher-rated songs
        weighted_profile = np.average(rated_features, axis=0, weights=weights)
        
        # Find similar songs using KNN
        distances, indices = self.knn.kneighbors([weighted_profile], n_neighbors=100)
        
        # Filter out songs that were already rated
        recommendations = []
        for idx in indices[0]:
            if idx not in rated_indices:
                recommendations.append(idx)
            if len(recommendations) >= n_recommendations:
                break
        
        # Get recommended songs
        recommended_songs = self.df.iloc[recommendations]
        
        return recommended_songs
    
    def display_recommendations(self, recommended_songs):
        """Display the recommended songs."""
        print("\n🎵 YOUR PERSONALIZED RECOMMENDATIONS 🎵\n")
        
        for idx, (_, song) in enumerate(recommended_songs.iterrows(), 1):
            self.display_song(song, idx)
        
        print("\n" + "="*70)
        print("Enjoy your personalized music recommendations!")
        print("="*70)
    
    def run(self):
        """Run the complete recommendation system."""
        # Get user ratings
        rated_songs, ratings = self.get_user_ratings()
        
        # Generate recommendations
        recommendations = self.recommend_songs(rated_songs, ratings, n_recommendations=10)
        
        # Display recommendations
        self.display_recommendations(recommendations)
        
        return recommendations


def main():
    """Main function to run the recommender system."""
    data_path = r"D:\codes\Datasets\spotify_data.csv"
    
    # Initialize and run the recommender
    recommender = SpotifyMusicRecommender(data_path)
    recommendations = recommender.run()
    
    # Optionally save recommendations to a file
    print("\nWould you like to save your recommendations? (yes/no): ", end='')
    save_choice = input().strip().lower()
    
    if save_choice in ['yes', 'y']:
        output_path = r'D:\codes\Datasets\recommendations.csv'
        recommendations.to_csv(output_path, index=False)
        print(f"\n✓ Recommendations saved to: recommendations.csv")


if __name__ == "__main__":
    main()
