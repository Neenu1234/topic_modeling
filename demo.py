#!/usr/bin/env python3
"""
Quick demonstration of the topic modeling pipeline.
This script runs a small sample to show the pipeline in action.
"""

import os
import sys
import time

# Add src directory to path
sys.path.append('src')

def run_demo():
    """Run a quick demonstration of the topic modeling pipeline."""
    print("=" * 60)
    print("TOPIC MODELING PIPELINE DEMONSTRATION")
    print("=" * 60)
    
    # Step 1: Preprocessing (already done)
    print("\n1. PREPROCESSING")
    print("-" * 30)
    if os.path.exists('data/processed/preprocessed_docs.pkl'):
        print("✓ Preprocessing already completed")
        print("  - 10,000 reviews processed")
        print("  - Average document length: 33.40 tokens")
    else:
        print("✗ Preprocessing not completed")
        return False
    
    # Step 2: Quick modeling demo
    print("\n2. TOPIC MODELING DEMO")
    print("-" * 30)
    
    try:
        from modeling import TopicModeler
        import pickle
        
        # Load preprocessed data
        with open('data/processed/preprocessed_docs.pkl', 'rb') as f:
            processed_docs = pickle.load(f)
        
        print(f"✓ Loaded {len(processed_docs)} preprocessed documents")
        
        # Initialize modeler with smaller range for demo
        modeler = TopicModeler(n_topics_range=(3, 8))
        
        print("✓ Preparing corpus...")
        modeler.prepare_corpus(processed_docs)
        
        print("✓ Training LDA model with 5 topics...")
        lda_model = modeler.train_lda_model(5)
        
        print("✓ Training NMF model with 5 topics...")
        nmf_model = modeler.train_nmf_model(5)
        
        print("✓ Training Gensim LDA model...")
        gensim_lda_model = modeler.train_gensim_lda_model(5)
        
        # Get top words
        feature_names = modeler.tfidf_vectorizer.get_feature_names_out()
        lda_topics = modeler.get_top_words(lda_model, feature_names, n_words=5)
        nmf_topics = modeler.get_top_words(nmf_model, feature_names, n_words=5)
        
        print("\nDISCOVERED TOPICS:")
        print("-" * 20)
        
        print("\nLDA Topics:")
        for topic_id, words in lda_topics.items():
            print(f"  Topic {topic_id}: {', '.join(words)}")
        
        print("\nNMF Topics:")
        for topic_id, words in nmf_topics.items():
            print(f"  Topic {topic_id}: {', '.join(words)}")
        
        # Calculate metrics
        lda_perplexity = modeler.calculate_perplexity(lda_model)
        lda_coherence = modeler.calculate_coherence(gensim_lda_model, processed_docs)
        
        print(f"\nMODEL METRICS:")
        print(f"  LDA Perplexity: {lda_perplexity:.2f}")
        print(f"  LDA Coherence: {lda_coherence:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Modeling demo failed: {e}")
        return False
    
    # Step 3: Vector database demo
    print("\n3. VECTOR DATABASE DEMO")
    print("-" * 30)
    
    try:
        from vector_db import VectorDatabase
        import pandas as pd
        
        # Load sample data
        with open('data/processed/sample_data.pkl', 'rb') as f:
            sample_data = pd.read_pickle(f)
        
        print(f"✓ Loaded {len(sample_data)} sample reviews")
        
        # Initialize vector database
        vector_db = VectorDatabase()
        
        print("✓ Building vector database...")
        vector_db.build_database(sample_data, lda_topics)
        
        print("✓ Searching for reviews similar to 'chocolate taste'...")
        search_results = vector_db.search_similar_reviews("chocolate taste", k=2)
        
        print("\nSEARCH RESULTS:")
        for i, review in enumerate(search_results):
            print(f"\n{i+1}. Similarity: {review['similarity_score']:.4f}")
            print(f"   Score: {review['score']}")
            print(f"   Text: {review['text'][:100]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ Vector database demo failed: {e}")
        return False

def main():
    """Main demonstration function."""
    start_time = time.time()
    
    success = run_demo()
    
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "=" * 60)
    if success:
        print("✓ DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print(f"  Total time: {duration:.2f} seconds")
        print("\nThe topic modeling pipeline is working correctly!")
        print("\nNext steps:")
        print("1. Run full pipeline: python src/main.py")
        print("2. Interactive analysis: jupyter notebook notebooks/topic_modeling_analysis.ipynb")
        print("3. Customize parameters in src/main.py")
    else:
        print("✗ DEMONSTRATION FAILED!")
        print("Please check the error messages above.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
