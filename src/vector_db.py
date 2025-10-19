"""
FAISS integration for retrieving representative reviews per topic.
Enables semantic search and similarity-based review retrieval.
"""

import numpy as np
import pandas as pd
import faiss
from typing import List, Dict, Tuple, Any
from sentence_transformers import SentenceTransformer
import pickle
import os
from tqdm import tqdm

class VectorDatabase:
    """Handles vector database operations using FAISS."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the vector database.
        
        Args:
            model_name: Name of the sentence transformer model
        """
        self.model_name = model_name
        self.encoder = SentenceTransformer(model_name)
        self.index = None
        self.reviews_df = None
        self.topic_vectors = None
        self.topic_results = None
        
    def encode_reviews(self, reviews: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode reviews into embeddings.
        
        Args:
            reviews: List of review texts
            batch_size: Batch size for encoding
            
        Returns:
            Array of embeddings
        """
        print(f"Encoding {len(reviews)} reviews using {self.model_name}...")
        
        embeddings = self.encoder.encode(
            reviews, 
            batch_size=batch_size, 
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        print(f"Generated embeddings shape: {embeddings.shape}")
        return embeddings
    
    def create_faiss_index(self, embeddings: np.ndarray, index_type: str = 'flat') -> faiss.Index:
        """
        Create FAISS index from embeddings.
        
        Args:
            embeddings: Review embeddings
            index_type: Type of FAISS index ('flat', 'ivf', 'hnsw')
            
        Returns:
            FAISS index
        """
        dimension = embeddings.shape[1]
        
        if index_type == 'flat':
            # Exact search index
            index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        elif index_type == 'ivf':
            # Inverted file index for faster search
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, 100)
            index.train(embeddings)
        elif index_type == 'hnsw':
            # Hierarchical Navigable Small World index
            index = faiss.IndexHNSWFlat(dimension, 32)
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        
        print(f"Created FAISS index with {index.ntotal} vectors")
        return index
    
    def encode_topics(self, topics: Dict[int, List[str]]) -> np.ndarray:
        """
        Encode topic words into embeddings.
        
        Args:
            topics: Dictionary mapping topic_id to list of top words
            
        Returns:
            Array of topic embeddings
        """
        print("Encoding topics...")
        
        topic_texts = []
        topic_ids = []
        
        for topic_id, words in topics.items():
            # Combine top words into a single text
            topic_text = ' '.join(words)
            topic_texts.append(topic_text)
            topic_ids.append(topic_id)
        
        topic_embeddings = self.encoder.encode(topic_texts, convert_to_numpy=True)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(topic_embeddings)
        
        return topic_embeddings, topic_ids
    
    def find_similar_reviews(self, query_embedding: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find most similar reviews to a query embedding.
        
        Args:
            query_embedding: Query embedding
            k: Number of similar reviews to return
            
        Returns:
            Tuple of (distances, indices)
        """
        if self.index is None:
            raise ValueError("FAISS index not created. Call create_faiss_index first.")
        
        # Normalize query embedding
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        # Search
        distances, indices = self.index.search(query_embedding, k)
        
        return distances[0], indices[0]
    
    def get_representative_reviews(self, topic_id: int, k: int = 5) -> List[Dict[str, Any]]:
        """
        Get representative reviews for a specific topic.
        
        Args:
            topic_id: Topic ID
            k: Number of representative reviews
            
        Returns:
            List of representative review dictionaries
        """
        if self.topic_vectors is None or self.topic_ids is None:
            raise ValueError("Topic vectors not encoded. Call encode_topics first.")
        
        # Find topic embedding
        topic_idx = self.topic_ids.index(topic_id)
        topic_embedding = self.topic_vectors[topic_idx]
        
        # Find similar reviews
        distances, indices = self.find_similar_reviews(topic_embedding, k)
        
        # Get review details
        representative_reviews = []
        for i, (distance, idx) in enumerate(zip(distances, indices)):
            if idx < len(self.reviews_df):
                review_data = self.reviews_df.iloc[idx].to_dict()
                review_data['similarity_score'] = float(distance)
                review_data['rank'] = i + 1
                representative_reviews.append(review_data)
        
        return representative_reviews
    
    def get_all_topic_representatives(self, k: int = 5) -> Dict[int, List[Dict[str, Any]]]:
        """
        Get representative reviews for all topics.
        
        Args:
            k: Number of representative reviews per topic
            
        Returns:
            Dictionary mapping topic_id to list of representative reviews
        """
        all_representatives = {}
        
        for topic_id in self.topic_ids:
            representatives = self.get_representative_reviews(topic_id, k)
            all_representatives[topic_id] = representatives
        
        return all_representatives
    
    def search_similar_reviews(self, query_text: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for reviews similar to a query text.
        
        Args:
            query_text: Query text
            k: Number of similar reviews
            
        Returns:
            List of similar review dictionaries
        """
        # Encode query
        query_embedding = self.encoder.encode([query_text], convert_to_numpy=True)
        
        # Find similar reviews
        distances, indices = self.find_similar_reviews(query_embedding[0], k)
        
        # Get review details
        similar_reviews = []
        for i, (distance, idx) in enumerate(zip(distances, indices)):
            if idx < len(self.reviews_df):
                review_data = self.reviews_df.iloc[idx].to_dict()
                review_data['similarity_score'] = float(distance)
                review_data['rank'] = i + 1
                similar_reviews.append(review_data)
        
        return similar_reviews
    
    def build_database(self, reviews_df: pd.DataFrame, topics: Dict[int, List[str]], 
                      index_type: str = 'flat') -> None:
        """
        Build the complete vector database.
        
        Args:
            reviews_df: DataFrame with review data
            topics: Dictionary mapping topic_id to list of top words
            index_type: Type of FAISS index
        """
        print("Building vector database...")
        
        # Store reviews dataframe
        self.reviews_df = reviews_df.copy()
        
        # Encode reviews
        review_texts = reviews_df['combined_text'].tolist()
        review_embeddings = self.encode_reviews(review_texts)
        
        # Create FAISS index
        self.index = self.create_faiss_index(review_embeddings, index_type)
        
        # Encode topics
        self.topic_vectors, self.topic_ids = self.encode_topics(topics)
        
        print("Vector database built successfully!")
    
    def save_database(self, output_dir: str) -> None:
        """Save the vector database to disk."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(output_dir, 'faiss_index.bin'))
        
        # Save other components
        with open(os.path.join(output_dir, 'reviews_df.pkl'), 'wb') as f:
            pickle.dump(self.reviews_df, f)
        
        with open(os.path.join(output_dir, 'topic_vectors.pkl'), 'wb') as f:
            pickle.dump(self.topic_vectors, f)
        
        with open(os.path.join(output_dir, 'topic_ids.pkl'), 'wb') as f:
            pickle.dump(self.topic_ids, f)
        
        print(f"Vector database saved to {output_dir}")
    
    def load_database(self, input_dir: str) -> None:
        """Load the vector database from disk."""
        # Load FAISS index
        self.index = faiss.read_index(os.path.join(input_dir, 'faiss_index.bin'))
        
        # Load other components
        with open(os.path.join(input_dir, 'reviews_df.pkl'), 'rb') as f:
            self.reviews_df = pickle.load(f)
        
        with open(os.path.join(input_dir, 'topic_vectors.pkl'), 'rb') as f:
            self.topic_vectors = pickle.load(f)
        
        with open(os.path.join(input_dir, 'topic_ids.pkl'), 'rb') as f:
            self.topic_ids = pickle.load(f)
        
        print(f"Vector database loaded from {input_dir}")

def main():
    """Main vector database pipeline."""
    import pickle
    
    # Load data
    with open('data/processed/sample_data.pkl', 'rb') as f:
        reviews_df = pickle.load(f)
    
    with open('results/models/topic_results.pkl', 'rb') as f:
        topic_results = pickle.load(f)
    
    # Initialize vector database
    vector_db = VectorDatabase()
    
    # Build database
    vector_db.build_database(reviews_df, topic_results['lda_topics'])
    
    # Get representative reviews for all topics
    print("\nGetting representative reviews for all topics...")
    all_representatives = vector_db.get_all_topic_representatives(k=3)
    
    # Display results
    for topic_id, representatives in all_representatives.items():
        print(f"\nTopic {topic_id} - Representative Reviews:")
        print(f"Top words: {', '.join(topic_results['lda_topics'][topic_id][:5])}")
        
        for i, review in enumerate(representatives):
            print(f"\n{i+1}. Similarity: {review['similarity_score']:.4f}")
            print(f"   Score: {review['score']}")
            print(f"   Text: {review['text'][:200]}...")
    
    # Save database
    vector_db.save_database('results/vector_db')
    
    # Example search
    print("\n\nExample search for 'chocolate taste':")
    search_results = vector_db.search_similar_reviews("chocolate taste", k=3)
    
    for i, review in enumerate(search_results):
        print(f"\n{i+1}. Similarity: {review['similarity_score']:.4f}")
        print(f"   Score: {review['score']}")
        print(f"   Text: {review['text'][:200]}...")
    
    print("\nVector database pipeline complete!")

if __name__ == "__main__":
    main()
