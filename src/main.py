"""
Main pipeline script for topic modeling with Amazon Fine Foods Reviews.
Orchestrates the complete workflow from preprocessing to visualization.
"""

import os
import sys
import pickle
import argparse
from typing import Optional

# Add src directory to path
sys.path.append('src')

from preprocessing import TextPreprocessor
from modeling import TopicModeler
from evaluation import TopicModelEvaluator
from visualization import TopicVisualizer
from vector_db import VectorDatabase

def run_preprocessing(data_path: str, sample_size: Optional[int] = None) -> None:
    """Run the preprocessing pipeline."""
    print("=" * 60)
    print("STEP 1: TEXT PREPROCESSING")
    print("=" * 60)
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    # Load data
    print("Loading Amazon Fine Foods Reviews dataset...")
    df = preprocessor.load_data(data_path)
    print(f"Loaded {len(df)} reviews")
    
    # Sample data if specified
    if sample_size and sample_size < len(df):
        df_sample = df.sample(n=sample_size, random_state=42)
        print(f"Using sample of {len(df_sample)} reviews")
    else:
        df_sample = df
    
    # Preprocess documents
    processed_docs = preprocessor.preprocess_documents(df_sample['combined_text'].tolist())
    
    # Save preprocessed data
    os.makedirs('data/processed', exist_ok=True)
    preprocessor.save_preprocessed_data(processed_docs, 'data/processed/preprocessed_docs.pkl')
    df_sample.to_pickle('data/processed/sample_data.pkl')
    
    print(f"Preprocessing complete! Processed {len(processed_docs)} documents")

def run_modeling(n_topics_range: tuple = (5, 15)) -> None:
    """Run the topic modeling pipeline."""
    print("=" * 60)
    print("STEP 2: TOPIC MODELING")
    print("=" * 60)
    
    # Load preprocessed data
    with open('data/processed/preprocessed_docs.pkl', 'rb') as f:
        processed_docs = pickle.load(f)
    
    # Initialize modeler
    modeler = TopicModeler(n_topics_range=n_topics_range)
    
    # Prepare corpus
    modeler.prepare_corpus(processed_docs)
    
    # Evaluate models
    evaluation_results = modeler.evaluate_models(processed_docs)
    
    # Find optimal number of topics
    import numpy as np
    best_idx = np.argmax(evaluation_results['lda_coherence'])
    optimal_topics = evaluation_results['topic_numbers'][best_idx]
    
    print(f"Optimal number of topics: {optimal_topics}")
    
    # Train final models
    modeler.train_final_models(optimal_topics, processed_docs)
    
    # Save models and results
    os.makedirs('results/models', exist_ok=True)
    modeler.save_models('results/models')
    
    # Save evaluation results
    with open('results/evaluation_results.pkl', 'wb') as f:
        pickle.dump(evaluation_results, f)
    
    print("Topic modeling complete!")

def run_evaluation() -> None:
    """Run the evaluation pipeline."""
    print("=" * 60)
    print("STEP 3: MODEL EVALUATION")
    print("=" * 60)
    
    # Load results
    with open('results/models/topic_results.pkl', 'rb') as f:
        topic_results = pickle.load(f)
    
    with open('results/evaluation_results.pkl', 'rb') as f:
        evaluation_results = pickle.load(f)
    
    # Load preprocessed data
    with open('data/processed/preprocessed_docs.pkl', 'rb') as f:
        processed_docs = pickle.load(f)
    
    # Load models
    with open('results/models/lda_model.pkl', 'rb') as f:
        lda_model = pickle.load(f)
    
    with open('results/models/nmf_model.pkl', 'rb') as f:
        nmf_model = pickle.load(f)
    
    with open('results/models/tfidf_vectorizer.pkl', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    
    # Initialize evaluator
    evaluator = TopicModelEvaluator()
    
    # Prepare data for evaluation
    text_docs = [' '.join(doc) for doc in processed_docs]
    tfidf_matrix = tfidf_vectorizer.transform(text_docs)
    
    # Create dictionary for coherence calculation
    from gensim.corpora import Dictionary
    dictionary = Dictionary(processed_docs)
    dictionary.filter_extremes(no_below=5, no_above=0.7)
    
    # Evaluate models
    lda_quality = evaluator.evaluate_topic_quality(
        topic_results['lda_topics'], lda_model, processed_docs, dictionary, tfidf_matrix
    )
    
    nmf_quality = evaluator.evaluate_topic_quality(
        topic_results['nmf_topics'], nmf_model, processed_docs, dictionary, tfidf_matrix
    )
    
    # Compare models
    comparison = evaluator.compare_models(lda_quality, nmf_quality)
    
    # Generate report
    report = evaluator.generate_evaluation_report(
        evaluation_results, lda_quality, nmf_quality, comparison
    )
    
    # Save results
    os.makedirs('results', exist_ok=True)
    evaluator.save_evaluation_results({
        'lda_quality': lda_quality,
        'nmf_quality': nmf_quality,
        'comparison': comparison,
        'report': report
    }, 'results/evaluation_detailed.pkl')
    
    # Save report to file
    with open('results/evaluation_report.txt', 'w') as f:
        f.write(report)
    
    # Print report
    print(report)
    
    print("Evaluation complete!")

def run_visualization() -> None:
    """Run the visualization pipeline."""
    print("=" * 60)
    print("STEP 4: VISUALIZATION")
    print("=" * 60)
    
    # Load results
    with open('results/models/topic_results.pkl', 'rb') as f:
        topic_results = pickle.load(f)
    
    with open('results/evaluation_results.pkl', 'rb') as f:
        evaluation_results = pickle.load(f)
    
    # Initialize visualizer
    visualizer = TopicVisualizer()
    
    # Generate comprehensive report
    visualizer.generate_comprehensive_report(
        topic_results, evaluation_results, 'results/visualizations'
    )
    
    print("Visualization complete!")

def run_vector_database() -> None:
    """Run the vector database pipeline."""
    print("=" * 60)
    print("STEP 5: VECTOR DATABASE")
    print("=" * 60)
    
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
    os.makedirs('results/vector_db', exist_ok=True)
    vector_db.save_database('results/vector_db')
    
    # Example search
    print("\n\nExample search for 'chocolate taste':")
    search_results = vector_db.search_similar_reviews("chocolate taste", k=3)
    
    for i, review in enumerate(search_results):
        print(f"\n{i+1}. Similarity: {review['similarity_score']:.4f}")
        print(f"   Score: {review['score']}")
        print(f"   Text: {review['text'][:200]}...")
    
    print("Vector database complete!")

def main():
    """Main pipeline function."""
    parser = argparse.ArgumentParser(description='Topic Modeling Pipeline for Amazon Fine Foods Reviews')
    parser.add_argument('--data-path', default='data/finefoods.txt', 
                       help='Path to the finefoods.txt file')
    parser.add_argument('--sample-size', type=int, default=10000,
                       help='Number of reviews to sample for processing')
    parser.add_argument('--n-topics-range', nargs=2, type=int, default=[5, 15],
                       help='Range of topic numbers to test (min max)')
    parser.add_argument('--skip-preprocessing', action='store_true',
                       help='Skip preprocessing step')
    parser.add_argument('--skip-modeling', action='store_true',
                       help='Skip modeling step')
    parser.add_argument('--skip-evaluation', action='store_true',
                       help='Skip evaluation step')
    parser.add_argument('--skip-visualization', action='store_true',
                       help='Skip visualization step')
    parser.add_argument('--skip-vector-db', action='store_true',
                       help='Skip vector database step')
    
    args = parser.parse_args()
    
    print("Amazon Fine Foods Reviews - Topic Modeling Pipeline")
    print("=" * 60)
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    try:
        # Step 1: Preprocessing
        if not args.skip_preprocessing:
            run_preprocessing(args.data_path, args.sample_size)
        else:
            print("Skipping preprocessing step...")
        
        # Step 2: Modeling
        if not args.skip_modeling:
            run_modeling(tuple(args.n_topics_range))
        else:
            print("Skipping modeling step...")
        
        # Step 3: Evaluation
        if not args.skip_evaluation:
            run_evaluation()
        else:
            print("Skipping evaluation step...")
        
        # Step 4: Visualization
        if not args.skip_visualization:
            run_visualization()
        else:
            print("Skipping visualization step...")
        
        # Step 5: Vector Database
        if not args.skip_vector_db:
            run_vector_database()
        else:
            print("Skipping vector database step...")
        
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE!")
        print("=" * 60)
        print("Results saved in the 'results' directory:")
        print("- models/: Trained topic models")
        print("- visualizations/: Plots and visualizations")
        print("- vector_db/: FAISS index and embeddings")
        print("- evaluation_report.txt: Detailed evaluation report")
        
    except Exception as e:
        print(f"Error in pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
