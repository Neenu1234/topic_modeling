"""
Visualization utilities for topic modeling results.
Includes topic word clouds, coherence plots, and interactive visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pyLDAvis
import pyLDAvis.gensim
from typing import Dict, List, Any
import pickle
import os

class TopicVisualizer:
    """Handles visualization of topic modeling results."""
    
    def __init__(self, figsize: tuple = (12, 8)):
        """
        Initialize the visualizer.
        
        Args:
            figsize: Default figure size for matplotlib plots
        """
        self.figsize = figsize
        plt.style.use('seaborn-v0_8')
        
    def plot_coherence_perplexity(self, evaluation_results: Dict[str, Any], 
                                save_path: str = None) -> None:
        """
        Plot coherence and perplexity vs number of topics.
        
        Args:
            evaluation_results: Results from topic number evaluation
            save_path: Path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        topic_numbers = evaluation_results['topic_numbers']
        
        # Coherence plot
        ax1.plot(topic_numbers, evaluation_results['lda_coherence'], 
                'bo-', linewidth=2, markersize=8, label='LDA Coherence')
        ax1.plot(topic_numbers, evaluation_results['nmf_coherence'], 
                'ro-', linewidth=2, markersize=8, label='NMF Coherence')
        ax1.set_xlabel('Number of Topics')
        ax1.set_ylabel('Coherence Score')
        ax1.set_title('Topic Coherence vs Number of Topics')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Perplexity plot
        ax2.plot(topic_numbers, evaluation_results['lda_perplexity'], 
                'go-', linewidth=2, markersize=8, label='LDA Perplexity')
        ax2.set_xlabel('Number of Topics')
        ax2.set_ylabel('Perplexity')
        ax2.set_title('LDA Perplexity vs Number of Topics')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Coherence/Perplexity plot saved to {save_path}")
        
        plt.show()
    
    def create_word_clouds(self, topics: Dict[int, List[str]], 
                         model_name: str, save_dir: str = None) -> None:
        """
        Create word clouds for each topic.
        
        Args:
            topics: Dictionary mapping topic_id to list of top words
            model_name: Name of the model (LDA/NMF)
            save_dir: Directory to save word clouds
        """
        n_topics = len(topics)
        cols = min(3, n_topics)
        rows = (n_topics + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if n_topics == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for topic_id, words in topics.items():
            # Create word cloud
            word_freq = {word: len(words) - i for i, word in enumerate(words)}
            wordcloud = WordCloud(
                width=400, height=300, 
                background_color='white',
                colormap='viridis',
                max_words=20
            ).generate_from_frequencies(word_freq)
            
            # Plot
            axes[topic_id].imshow(wordcloud, interpolation='bilinear')
            axes[topic_id].set_title(f'Topic {topic_id}', fontsize=14, fontweight='bold')
            axes[topic_id].axis('off')
        
        # Hide unused subplots
        for i in range(n_topics, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'{model_name} Topic Word Clouds', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'{model_name.lower()}_word_clouds.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Word clouds saved to {save_path}")
        
        plt.show()
    
    def plot_topic_distribution(self, topic_results: Dict[str, Any], 
                              save_path: str = None) -> None:
        """
        Plot topic distribution across documents.
        
        Args:
            topic_results: Results from topic modeling
            save_path: Path to save the plot
        """
        # This would need document-topic probabilities
        # For now, create a placeholder plot
        fig, ax = plt.subplots(figsize=self.figsize)
        
        n_topics = topic_results['n_topics']
        topic_ids = list(range(n_topics))
        
        # Simulate topic distribution (in real implementation, use actual data)
        topic_counts = np.random.poisson(100, n_topics)
        
        bars = ax.bar(topic_ids, topic_counts, color='skyblue', alpha=0.7)
        ax.set_xlabel('Topic ID')
        ax.set_ylabel('Number of Documents')
        ax.set_title('Topic Distribution Across Documents')
        ax.set_xticks(topic_ids)
        
        # Add value labels on bars
        for bar, count in zip(bars, topic_counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Topic distribution plot saved to {save_path}")
        
        plt.show()
    
    def create_interactive_topic_plot(self, topics: Dict[int, List[str]], 
                                    model_name: str, save_path: str = None) -> None:
        """
        Create interactive plotly visualization of topics.
        
        Args:
            topics: Dictionary mapping topic_id to list of top words
            model_name: Name of the model
            save_path: Path to save the HTML file
        """
        # Prepare data for plotting
        plot_data = []
        for topic_id, words in topics.items():
            for i, word in enumerate(words):
                plot_data.append({
                    'Topic': f'Topic {topic_id}',
                    'Word': word,
                    'Rank': i + 1,
                    'Score': len(words) - i  # Higher score for higher ranked words
                })
        
        df = pd.DataFrame(plot_data)
        
        # Create interactive plot
        fig = px.scatter(df, x='Rank', y='Score', color='Topic',
                        hover_data=['Word'], title=f'{model_name} Topics - Word Rankings')
        
        fig.update_layout(
            xaxis_title='Word Rank',
            yaxis_title='Word Score',
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"Interactive plot saved to {save_path}")
        
        fig.show()
    
    def create_topic_comparison_plot(self, lda_topics: Dict[int, List[str]], 
                                   nmf_topics: Dict[int, List[str]], 
                                   save_path: str = None) -> None:
        """
        Create comparison plot between LDA and NMF topics.
        
        Args:
            lda_topics: LDA topic words
            nmf_topics: NMF topic words
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 1, figsize=(15, 12))
        
        # LDA topics
        lda_words = []
        lda_topics_list = []
        for topic_id, words in lda_topics.items():
            for word in words[:5]:  # Top 5 words
                lda_words.append(word)
                lda_topics_list.append(f'LDA Topic {topic_id}')
        
        # NMF topics
        nmf_words = []
        nmf_topics_list = []
        for topic_id, words in nmf_topics.items():
            for word in words[:5]:  # Top 5 words
                nmf_words.append(word)
                nmf_topics_list.append(f'NMF Topic {topic_id}')
        
        # Create word clouds
        if lda_words:
            lda_wordcloud = WordCloud(width=600, height=300, background_color='white').generate(' '.join(lda_words))
            axes[0].imshow(lda_wordcloud, interpolation='bilinear')
            axes[0].set_title('LDA Topics', fontsize=14, fontweight='bold')
            axes[0].axis('off')
        
        if nmf_words:
            nmf_wordcloud = WordCloud(width=600, height=300, background_color='white').generate(' '.join(nmf_words))
            axes[1].imshow(nmf_wordcloud, interpolation='bilinear')
            axes[1].set_title('NMF Topics', fontsize=14, fontweight='bold')
            axes[1].axis('off')
        
        plt.suptitle('LDA vs NMF Topic Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Topic comparison plot saved to {save_path}")
        
        plt.show()
    
    def create_pyldavis_visualization(self, gensim_model, corpus, dictionary, 
                                    save_path: str = None) -> None:
        """
        Create pyLDAvis interactive visualization.
        
        Args:
            gensim_model: Trained Gensim LDA model
            corpus: Gensim corpus
            dictionary: Gensim dictionary
            save_path: Path to save the HTML file
        """
        try:
            vis = pyLDAvis.gensim.prepare(gensim_model, corpus, dictionary, sort_topics=False)
            
            if save_path:
                pyLDAvis.save_html(vis, save_path)
                print(f"pyLDAvis visualization saved to {save_path}")
            
            pyLDAvis.display(vis)
            
        except Exception as e:
            print(f"Error creating pyLDAvis visualization: {e}")
    
    def generate_comprehensive_report(self, topic_results: Dict[str, Any], 
                                    evaluation_results: Dict[str, Any],
                                    save_dir: str = None) -> None:
        """
        Generate comprehensive visualization report.
        
        Args:
            topic_results: Topic modeling results
            evaluation_results: Evaluation results
            save_dir: Directory to save all visualizations
        """
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        print("Generating comprehensive visualization report...")
        
        # 1. Coherence and Perplexity plots
        coherence_path = os.path.join(save_dir, 'coherence_perplexity.png') if save_dir else None
        self.plot_coherence_perplexity(evaluation_results, coherence_path)
        
        # 2. LDA Word Clouds
        lda_clouds_path = os.path.join(save_dir, 'lda_word_clouds') if save_dir else None
        self.create_word_clouds(topic_results['lda_topics'], 'LDA', lda_clouds_path)
        
        # 3. NMF Word Clouds
        nmf_clouds_path = os.path.join(save_dir, 'nmf_word_clouds') if save_dir else None
        self.create_word_clouds(topic_results['nmf_topics'], 'NMF', nmf_clouds_path)
        
        # 4. Topic Comparison
        comparison_path = os.path.join(save_dir, 'topic_comparison.png') if save_dir else None
        self.create_topic_comparison_plot(
            topic_results['lda_topics'], 
            topic_results['nmf_topics'], 
            comparison_path
        )
        
        # 5. Interactive plots
        lda_interactive_path = os.path.join(save_dir, 'lda_interactive.html') if save_dir else None
        self.create_interactive_topic_plot(topic_results['lda_topics'], 'LDA', lda_interactive_path)
        
        nmf_interactive_path = os.path.join(save_dir, 'nmf_interactive.html') if save_dir else None
        self.create_interactive_topic_plot(topic_results['nmf_topics'], 'NMF', nmf_interactive_path)
        
        print("Comprehensive visualization report generated!")

def main():
    """Main visualization pipeline."""
    import pickle
    
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
    
    print("Visualization pipeline complete!")

if __name__ == "__main__":
    main()
