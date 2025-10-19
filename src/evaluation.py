"""
Model evaluation utilities for topic modeling.
Includes coherence, perplexity, and topic quality metrics.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from sklearn.metrics import silhouette_score
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

class TopicModelEvaluator:
    """Handles evaluation of topic models."""
    
    def __init__(self):
        """Initialize the evaluator."""
        self.evaluation_results = {}
        
    def calculate_topic_diversity(self, topics: Dict[int, List[str]]) -> float:
        """
        Calculate topic diversity (average pairwise Jaccard distance).
        
        Args:
            topics: Dictionary mapping topic_id to list of top words
            
        Returns:
            Topic diversity score
        """
        topic_words = list(topics.values())
        n_topics = len(topic_words)
        
        if n_topics < 2:
            return 0.0
        
        jaccard_distances = []
        for i in range(n_topics):
            for j in range(i + 1, n_topics):
                set1 = set(topic_words[i])
                set2 = set(topic_words[j])
                
                intersection = len(set1.intersection(set2))
                union = len(set1.union(set2))
                
                jaccard_similarity = intersection / union if union > 0 else 0
                jaccard_distance = 1 - jaccard_similarity
                jaccard_distances.append(jaccard_distance)
        
        return np.mean(jaccard_distances)
    
    def calculate_topic_coherence_detailed(self, model, processed_docs: List[List[str]], 
                                         dictionary, n_words: int = 10) -> Dict[str, float]:
        """
        Calculate detailed coherence metrics.
        
        Args:
            model: Trained topic model
            processed_docs: Original processed documents
            dictionary: Gensim dictionary
            n_words: Number of top words to consider
            
        Returns:
            Dictionary with different coherence metrics
        """
        coherence_metrics = {}
        
        # Different coherence measures
        measures = ['u_mass', 'c_v', 'c_uci', 'c_npmi']
        
        for measure in measures:
            try:
                coherence_model = CoherenceModel(
                    model=model,
                    texts=processed_docs,
                    dictionary=dictionary,
                    coherence=measure,
                    topn=n_words
                )
                coherence_metrics[measure] = coherence_model.get_coherence()
            except Exception as e:
                print(f"Could not calculate {measure} coherence: {e}")
                coherence_metrics[measure] = 0.0
        
        return coherence_metrics
    
    def analyze_topic_distribution(self, model, tfidf_matrix) -> Dict[str, Any]:
        """
        Analyze topic distribution across documents.
        
        Args:
            model: Trained topic model
            tfidf_matrix: TF-IDF matrix
            
        Returns:
            Dictionary with distribution statistics
        """
        # Get document-topic probabilities
        doc_topic_probs = model.transform(tfidf_matrix)
        
        # Calculate statistics
        stats = {
            'avg_topic_prob': np.mean(doc_topic_probs, axis=0),
            'topic_entropy': -np.sum(doc_topic_probs * np.log(doc_topic_probs + 1e-10), axis=1),
            'dominant_topic': np.argmax(doc_topic_probs, axis=1),
            'max_topic_prob': np.max(doc_topic_probs, axis=1),
            'topic_counts': np.bincount(np.argmax(doc_topic_probs, axis=1))
        }
        
        return stats
    
    def evaluate_topic_quality(self, topics: Dict[int, List[str]], 
                             model, processed_docs: List[List[str]], 
                             dictionary, tfidf_matrix) -> Dict[str, Any]:
        """
        Comprehensive topic quality evaluation.
        
        Args:
            topics: Dictionary mapping topic_id to list of top words
            model: Trained topic model
            processed_docs: Original processed documents
            dictionary: Gensim dictionary
            tfidf_matrix: TF-IDF matrix
            
        Returns:
            Dictionary with quality metrics
        """
        print("Evaluating topic quality...")
        
        # Basic metrics
        n_topics = len(topics)
        avg_words_per_topic = np.mean([len(words) for words in topics.values()])
        
        # Topic diversity
        diversity = self.calculate_topic_diversity(topics)
        
        # Detailed coherence
        coherence_metrics = self.calculate_topic_coherence_detailed(
            model, processed_docs, dictionary
        )
        
        # Topic distribution analysis
        distribution_stats = self.analyze_topic_distribution(model, tfidf_matrix)
        
        # Calculate topic balance (how evenly distributed topics are)
        topic_counts = distribution_stats['topic_counts']
        topic_balance = 1 - (np.std(topic_counts) / np.mean(topic_counts)) if np.mean(topic_counts) > 0 else 0
        
        quality_metrics = {
            'n_topics': n_topics,
            'avg_words_per_topic': avg_words_per_topic,
            'topic_diversity': diversity,
            'topic_balance': topic_balance,
            'coherence_metrics': coherence_metrics,
            'distribution_stats': distribution_stats,
            'avg_topic_prob': np.mean(distribution_stats['avg_topic_prob']),
            'avg_entropy': np.mean(distribution_stats['topic_entropy']),
            'avg_max_prob': np.mean(distribution_stats['max_topic_prob'])
        }
        
        return quality_metrics
    
    def compare_models(self, lda_results: Dict[str, Any], nmf_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare LDA and NMF models.
        
        Args:
            lda_results: LDA model results
            nmf_results: NMF model results
            
        Returns:
            Comparison results
        """
        comparison = {
            'lda_coherence': lda_results.get('lda_coherence', 0),
            'nmf_coherence': nmf_results.get('lda_coherence', 0),  # Using same metric
            'lda_perplexity': lda_results.get('lda_perplexity', 0),
            'lda_diversity': lda_results.get('topic_diversity', 0),
            'nmf_diversity': nmf_results.get('topic_diversity', 0),
            'lda_balance': lda_results.get('topic_balance', 0),
            'nmf_balance': nmf_results.get('topic_balance', 0)
        }
        
        # Determine winner
        if comparison['lda_coherence'] > comparison['nmf_coherence']:
            comparison['coherence_winner'] = 'LDA'
        else:
            comparison['coherence_winner'] = 'NMF'
        
        if comparison['lda_diversity'] > comparison['nmf_diversity']:
            comparison['diversity_winner'] = 'LDA'
        else:
            comparison['diversity_winner'] = 'NMF'
        
        return comparison
    
    def generate_evaluation_report(self, evaluation_results: Dict[str, Any], 
                                lda_quality: Dict[str, Any], 
                                nmf_quality: Dict[str, Any],
                                comparison: Dict[str, Any]) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            evaluation_results: Topic number evaluation results
            lda_quality: LDA quality metrics
            nmf_quality: NMF quality metrics
            comparison: Model comparison results
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 60)
        report.append("TOPIC MODELING EVALUATION REPORT")
        report.append("=" * 60)
        
        # Topic number analysis
        report.append("\n1. TOPIC NUMBER ANALYSIS")
        report.append("-" * 30)
        best_idx = np.argmax(evaluation_results['lda_coherence'])
        optimal_topics = evaluation_results['topic_numbers'][best_idx]
        report.append(f"Optimal number of topics: {optimal_topics}")
        report.append(f"Best coherence score: {evaluation_results['lda_coherence'][best_idx]:.4f}")
        
        # LDA Results
        report.append("\n2. LDA MODEL RESULTS")
        report.append("-" * 30)
        report.append(f"Perplexity: {lda_quality.get('lda_perplexity', 'N/A')}")
        report.append(f"Topic Diversity: {lda_quality.get('topic_diversity', 0):.4f}")
        report.append(f"Topic Balance: {lda_quality.get('topic_balance', 0):.4f}")
        report.append(f"Average Entropy: {lda_quality.get('avg_entropy', 0):.4f}")
        
        if 'coherence_metrics' in lda_quality:
            report.append("Coherence Metrics:")
            for metric, value in lda_quality['coherence_metrics'].items():
                report.append(f"  {metric}: {value:.4f}")
        
        # NMF Results
        report.append("\n3. NMF MODEL RESULTS")
        report.append("-" * 30)
        report.append(f"Topic Diversity: {nmf_quality.get('topic_diversity', 0):.4f}")
        report.append(f"Topic Balance: {nmf_quality.get('topic_balance', 0):.4f}")
        report.append(f"Average Entropy: {nmf_quality.get('avg_entropy', 0):.4f}")
        
        # Model Comparison
        report.append("\n4. MODEL COMPARISON")
        report.append("-" * 30)
        report.append(f"Coherence Winner: {comparison.get('coherence_winner', 'N/A')}")
        report.append(f"Diversity Winner: {comparison.get('diversity_winner', 'N/A')}")
        report.append(f"LDA Coherence: {comparison.get('lda_coherence', 0):.4f}")
        report.append(f"NMF Coherence: {comparison.get('nmf_coherence', 0):.4f}")
        
        # Recommendations
        report.append("\n5. RECOMMENDATIONS")
        report.append("-" * 30)
        if comparison.get('coherence_winner') == 'LDA':
            report.append("• LDA shows better coherence and is recommended for interpretability")
        else:
            report.append("• NMF shows better coherence and is recommended for interpretability")
        
        if comparison.get('diversity_winner') == 'LDA':
            report.append("• LDA produces more diverse topics")
        else:
            report.append("• NMF produces more diverse topics")
        
        report.append(f"• Optimal topic count: {optimal_topics}")
        report.append("• Consider using both models for different analysis purposes")
        
        return "\n".join(report)
    
    def save_evaluation_results(self, results: Dict[str, Any], output_path: str):
        """Save evaluation results to file."""
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"Evaluation results saved to {output_path}")

def main():
    """Main evaluation pipeline."""
    import pickle
    
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
    
    # Evaluate LDA
    lda_quality = evaluator.evaluate_topic_quality(
        topic_results['lda_topics'], lda_model, processed_docs, dictionary, tfidf_matrix
    )
    
    # Evaluate NMF
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
    evaluator.save_evaluation_results({
        'lda_quality': lda_quality,
        'nmf_quality': nmf_quality,
        'comparison': comparison,
        'report': report
    }, 'results/evaluation_detailed.pkl')
    
    # Print report
    print(report)
    
    # Save report to file
    with open('results/evaluation_report.txt', 'w') as f:
        f.write(report)
    
    print("\nDetailed evaluation complete!")

if __name__ == "__main__":
    main()
