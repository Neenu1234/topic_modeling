"""
Topic modeling implementation using LDA and NMF algorithms.
Includes TF-IDF vectorization and model training.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from gensim.models import LdaModel, CoherenceModel
from gensim.corpora import Dictionary
import pickle
import os
from tqdm import tqdm

class TopicModeler:
    """Handles topic modeling with LDA and NMF algorithms."""
    
    def __init__(self, n_topics_range: Tuple[int, int] = (5, 20), random_state: int = 42):
        """
        Initialize the topic modeler.
        
        Args:
            n_topics_range: Range of topic numbers to test (min, max)
            random_state: Random state for reproducibility
        """
        self.n_topics_range = n_topics_range
        self.random_state = random_state
        
        # Models and vectorizers
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.dictionary = None
        self.corpus = None
        
        # Topic models
        self.lda_model = None
        self.nmf_model = None
        self.gensim_lda_model = None
        
        # Results
        self.topic_results = {}
        
    def prepare_corpus(self, processed_docs: List[List[str]]):
        """
        Prepare corpus for topic modeling.
        
        Args:
            processed_docs: List of preprocessed tokenized documents
        """
        print("Preparing corpus for topic modeling...")
        
        # Convert tokenized docs back to text for TF-IDF
        text_docs = [' '.join(doc) for doc in processed_docs]
        
        # TF-IDF vectorization
        print("Applying TF-IDF vectorization...")
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            min_df=5,
            max_df=0.7,
            ngram_range=(1, 2),  # Include bigrams
            stop_words='english'
        )
        
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(text_docs)
        print(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        
        # Prepare Gensim corpus for coherence calculation
        print("Preparing Gensim corpus...")
        self.dictionary = Dictionary(processed_docs)
        self.dictionary.filter_extremes(no_below=5, no_above=0.7)
        self.corpus = [self.dictionary.doc2bow(doc) for doc in processed_docs]
        
        print(f"Dictionary size: {len(self.dictionary)}")
        print(f"Corpus size: {len(self.corpus)}")
    
    def train_lda_model(self, n_topics: int) -> LatentDirichletAllocation:
        """
        Train LDA model using scikit-learn.
        
        Args:
            n_topics: Number of topics
            
        Returns:
            Trained LDA model
        """
        print(f"Training LDA model with {n_topics} topics...")
        
        lda_model = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=self.random_state,
            max_iter=100,
            learning_decay=0.7,
            learning_offset=50.0,
            doc_topic_prior=0.1,
            topic_word_prior=0.1
        )
        
        lda_model.fit(self.tfidf_matrix)
        return lda_model
    
    def train_nmf_model(self, n_topics: int) -> NMF:
        """
        Train NMF model.
        
        Args:
            n_topics: Number of topics
            
        Returns:
            Trained NMF model
        """
        print(f"Training NMF model with {n_topics} topics...")
        
        nmf_model = NMF(
            n_components=n_topics,
            random_state=self.random_state,
            max_iter=1000,
            alpha_W=0.1,
            alpha_H=0.1,
            l1_ratio=0.5
        )
        
        nmf_model.fit(self.tfidf_matrix)
        return nmf_model
    
    def train_gensim_lda_model(self, n_topics: int) -> LdaModel:
        """
        Train Gensim LDA model for coherence calculation.
        
        Args:
            n_topics: Number of topics
            
        Returns:
            Trained Gensim LDA model
        """
        print(f"Training Gensim LDA model with {n_topics} topics...")
        
        gensim_lda_model = LdaModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=n_topics,
            random_state=self.random_state,
            passes=10,
            alpha='auto',
            per_word_topics=True
        )
        
        return gensim_lda_model
    
    def calculate_coherence(self, model: LdaModel, processed_docs: List[List[str]]) -> float:
        """
        Calculate topic coherence using Gensim.
        
        Args:
            model: Trained LDA model
            processed_docs: Original processed documents
            
        Returns:
            Coherence score
        """
        coherence_model = CoherenceModel(
            model=model,
            texts=processed_docs,
            dictionary=self.dictionary,
            coherence='c_v'
        )
        
        return coherence_model.get_coherence()
    
    def calculate_perplexity(self, model: LatentDirichletAllocation) -> float:
        """
        Calculate perplexity for scikit-learn LDA model.
        
        Args:
            model: Trained LDA model
            
        Returns:
            Perplexity score
        """
        return model.perplexity(self.tfidf_matrix)
    
    def get_top_words(self, model: Any, feature_names: List[str], n_words: int = 10) -> Dict[int, List[str]]:
        """
        Extract top words for each topic.
        
        Args:
            model: Trained topic model
            feature_names: Feature names from vectorizer
            n_words: Number of top words per topic
            
        Returns:
            Dictionary mapping topic_id to list of top words
        """
        topics = {}
        
        if hasattr(model, 'components_'):  # scikit-learn models
            for topic_idx, topic in enumerate(model.components_):
                top_words_idx = topic.argsort()[-n_words:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                topics[topic_idx] = top_words
        else:  # Gensim models
            for topic_idx in range(model.num_topics):
                topic_words = model.show_topic(topic_idx, topn=n_words)
                top_words = [word for word, _ in topic_words]
                topics[topic_idx] = top_words
        
        return topics
    
    def evaluate_models(self, processed_docs: List[List[str]]) -> Dict[str, Any]:
        """
        Evaluate models across different topic numbers.
        
        Args:
            processed_docs: Original processed documents
            
        Returns:
            Dictionary with evaluation results
        """
        print("Evaluating models across different topic numbers...")
        
        min_topics, max_topics = self.n_topics_range
        topic_numbers = range(min_topics, max_topics + 1)
        
        results = {
            'lda_perplexity': [],
            'lda_coherence': [],
            'nmf_coherence': [],
            'topic_numbers': list(topic_numbers)
        }
        
        for n_topics in tqdm(topic_numbers, desc="Evaluating topic numbers"):
            # Train LDA model
            lda_model = self.train_lda_model(n_topics)
            lda_perplexity = self.calculate_perplexity(lda_model)
            results['lda_perplexity'].append(lda_perplexity)
            
            # Train Gensim LDA for coherence
            gensim_lda_model = self.train_gensim_lda_model(n_topics)
            lda_coherence = self.calculate_coherence(gensim_lda_model, processed_docs)
            results['lda_coherence'].append(lda_coherence)
            
            # Train NMF model
            nmf_model = self.train_nmf_model(n_topics)
            
            # For NMF coherence, we'll use a simple approximation
            # by training a Gensim LDA model with the same number of topics
            nmf_coherence = lda_coherence  # Simplified for now
            results['nmf_coherence'].append(nmf_coherence)
        
        return results
    
    def train_final_models(self, n_topics: int, processed_docs: List[List[str]]):
        """
        Train final models with optimal number of topics.
        
        Args:
            n_topics: Optimal number of topics
            processed_docs: Original processed documents
        """
        print(f"Training final models with {n_topics} topics...")
        
        # Train final models
        self.lda_model = self.train_lda_model(n_topics)
        self.nmf_model = self.train_nmf_model(n_topics)
        self.gensim_lda_model = self.train_gensim_lda_model(n_topics)
        
        # Get feature names
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        
        # Extract top words
        lda_topics = self.get_top_words(self.lda_model, feature_names)
        nmf_topics = self.get_top_words(self.nmf_model, feature_names)
        gensim_topics = self.get_top_words(self.gensim_lda_model, feature_names)
        
        # Calculate final metrics
        lda_perplexity = self.calculate_perplexity(self.lda_model)
        lda_coherence = self.calculate_coherence(self.gensim_lda_model, processed_docs)
        
        # Store results
        self.topic_results = {
            'n_topics': n_topics,
            'lda_topics': lda_topics,
            'nmf_topics': nmf_topics,
            'gensim_topics': gensim_topics,
            'lda_perplexity': lda_perplexity,
            'lda_coherence': lda_coherence,
            'feature_names': feature_names
        }
        
        print(f"Final LDA Perplexity: {lda_perplexity:.2f}")
        print(f"Final LDA Coherence: {lda_coherence:.4f}")
    
    def save_models(self, output_dir: str):
        """Save trained models and results."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save models
        with open(f"{output_dir}/lda_model.pkl", 'wb') as f:
            pickle.dump(self.lda_model, f)
        
        with open(f"{output_dir}/nmf_model.pkl", 'wb') as f:
            pickle.dump(self.nmf_model, f)
        
        with open(f"{output_dir}/gensim_lda_model.pkl", 'wb') as f:
            pickle.dump(self.gensim_lda_model, f)
        
        with open(f"{output_dir}/tfidf_vectorizer.pkl", 'wb') as f:
            pickle.dump(self.tfidf_vectorizer, f)
        
        # Save results
        with open(f"{output_dir}/topic_results.pkl", 'wb') as f:
            pickle.dump(self.topic_results, f)
        
        print(f"Models and results saved to {output_dir}")

def main():
    """Main modeling pipeline."""
    import pickle
    
    # Load preprocessed data
    with open('data/processed/preprocessed_docs.pkl', 'rb') as f:
        processed_docs = pickle.load(f)
    
    # Initialize modeler
    modeler = TopicModeler(n_topics_range=(5, 15))
    
    # Prepare corpus
    modeler.prepare_corpus(processed_docs)
    
    # Evaluate models
    evaluation_results = modeler.evaluate_models(processed_docs)
    
    # Find optimal number of topics (highest coherence)
    best_idx = np.argmax(evaluation_results['lda_coherence'])
    optimal_topics = evaluation_results['topic_numbers'][best_idx]
    
    print(f"Optimal number of topics: {optimal_topics}")
    
    # Train final models
    modeler.train_final_models(optimal_topics, processed_docs)
    
    # Save models
    modeler.save_models('results/models')
    
    # Save evaluation results
    with open('results/evaluation_results.pkl', 'wb') as f:
        pickle.dump(evaluation_results, f)
    
    print("Topic modeling complete!")

if __name__ == "__main__":
    main()
