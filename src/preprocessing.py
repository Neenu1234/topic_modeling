"""
Text preprocessing pipeline for Amazon Fine Foods Reviews dataset.
Handles data loading, cleaning, tokenization, and phrase extraction.
"""

import pandas as pd
import numpy as np
import re
import string
from typing import List, Tuple, Dict
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models import Phrases
from gensim.models.phrases import Phraser
import pickle
import os

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class TextPreprocessor:
    """Handles text preprocessing for topic modeling."""
    
    def __init__(self, min_word_length: int = 3, min_bigram_count: int = 5, min_trigram_count: int = 5):
        """
        Initialize the preprocessor.
        
        Args:
            min_word_length: Minimum length for words to be kept
            min_bigram_count: Minimum count for bigrams to be kept
            min_trigram_count: Minimum count for trigrams to be kept
        """
        self.min_word_length = min_word_length
        self.min_bigram_count = min_bigram_count
        self.min_trigram_count = min_trigram_count
        
        # Initialize components
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Add common food-related stopwords
        self.stop_words.update([
            'food', 'product', 'amazon', 'review', 'buy', 'purchase', 
            'order', 'delivery', 'shipping', 'price', 'cost', 'money',
            'time', 'day', 'week', 'month', 'year', 'good', 'bad',
            'great', 'excellent', 'terrible', 'awful', 'okay', 'ok',
            'like', 'love', 'hate', 'dislike', 'recommend', 'would',
            'will', 'should', 'could', 'might', 'may', 'can', 'get',
            'got', 'going', 'come', 'came', 'make', 'made', 'take',
            'took', 'give', 'gave', 'see', 'saw', 'know', 'knew'
        ])
        
        # Initialize phrase models
        self.bigram_model = None
        self.trigram_model = None
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load Amazon Fine Foods Reviews dataset.
        
        Args:
            file_path: Path to the finefoods.txt file
            
        Returns:
            DataFrame with parsed reviews
        """
        reviews = []
        current_review = {}
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            for line in file:
                line = line.strip()
                
                if line.startswith('product/productId:'):
                    if current_review:  # Save previous review
                        reviews.append(current_review)
                    current_review = {'productId': line.split(':', 1)[1].strip()}
                    
                elif line.startswith('review/userId:'):
                    current_review['userId'] = line.split(':', 1)[1].strip()
                    
                elif line.startswith('review/profileName:'):
                    current_review['profileName'] = line.split(':', 1)[1].strip()
                    
                elif line.startswith('review/helpfulness:'):
                    current_review['helpfulness'] = line.split(':', 1)[1].strip()
                    
                elif line.startswith('review/score:'):
                    current_review['score'] = float(line.split(':', 1)[1].strip())
                    
                elif line.startswith('review/time:'):
                    current_review['time'] = int(line.split(':', 1)[1].strip())
                    
                elif line.startswith('review/summary:'):
                    current_review['summary'] = line.split(':', 1)[1].strip()
                    
                elif line.startswith('review/text:'):
                    current_review['text'] = line.split(':', 1)[1].strip()
                    
                elif line == '' and current_review:  # Empty line indicates end of review
                    reviews.append(current_review)
                    current_review = {}
        
        # Add the last review
        if current_review:
            reviews.append(current_review)
            
        df = pd.DataFrame(reviews)
        
        # Clean up the data
        df = df.dropna(subset=['text'])  # Remove reviews without text
        df['text'] = df['text'].astype(str)
        df['summary'] = df['summary'].fillna('').astype(str)
        
        # Combine summary and text for better topic modeling
        df['combined_text'] = df['summary'] + ' ' + df['text']
        
        return df
    
    def clean_text(self, text: str) -> str:
        """
        Clean text by removing unwanted characters and normalizing.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def tokenize_and_lemmatize(self, text: str) -> List[str]:
        """
        Tokenize text and lemmatize tokens.
        
        Args:
            text: Input text
            
        Returns:
            List of lemmatized tokens
        """
        # Tokenize
        tokens = word_tokenize(text)
        
        # Filter and lemmatize
        processed_tokens = []
        for token in tokens:
            # Remove punctuation and short words
            if token.isalpha() and len(token) >= self.min_word_length:
                # Lemmatize
                lemmatized = self.lemmatizer.lemmatize(token)
                # Remove stopwords
                if lemmatized not in self.stop_words:
                    processed_tokens.append(lemmatized)
        
        return processed_tokens
    
    def extract_phrases(self, tokenized_docs: List[List[str]]) -> Tuple[List[List[str]], Phrases, Phrases]:
        """
        Extract bigrams and trigrams from tokenized documents.
        
        Args:
            tokenized_docs: List of tokenized documents
            
        Returns:
            Tuple of (processed_docs, bigram_model, trigram_model)
        """
        # Train bigram model
        bigram_model = Phrases(tokenized_docs, min_count=self.min_bigram_count, threshold=10)
        bigram_phraser = Phraser(bigram_model)
        
        # Apply bigrams
        bigram_docs = [bigram_phraser[doc] for doc in tokenized_docs]
        
        # Train trigram model
        trigram_model = Phrases(bigram_docs, min_count=self.min_trigram_count, threshold=10)
        trigram_phraser = Phraser(trigram_model)
        
        # Apply trigrams
        trigram_docs = [trigram_phraser[doc] for doc in bigram_docs]
        
        # Store models
        self.bigram_model = bigram_phraser
        self.trigram_model = trigram_phraser
        
        return trigram_docs, bigram_phraser, trigram_phraser
    
    def preprocess_documents(self, texts: List[str]) -> List[List[str]]:
        """
        Complete preprocessing pipeline for a list of texts.
        
        Args:
            texts: List of raw texts
            
        Returns:
            List of preprocessed tokenized documents
        """
        print("Cleaning texts...")
        cleaned_texts = [self.clean_text(text) for text in texts]
        
        print("Tokenizing and lemmatizing...")
        tokenized_docs = [self.tokenize_and_lemmatize(text) for text in cleaned_texts]
        
        # Filter out empty documents
        tokenized_docs = [doc for doc in tokenized_docs if len(doc) > 0]
        
        print("Extracting phrases...")
        processed_docs, _, _ = self.extract_phrases(tokenized_docs)
        
        return processed_docs
    
    def save_preprocessed_data(self, processed_docs: List[List[str]], output_path: str):
        """Save preprocessed data to file."""
        with open(output_path, 'wb') as f:
            pickle.dump(processed_docs, f)
        print(f"Preprocessed data saved to {output_path}")
    
    def load_preprocessed_data(self, input_path: str) -> List[List[str]]:
        """Load preprocessed data from file."""
        with open(input_path, 'rb') as f:
            processed_docs = pickle.load(f)
        print(f"Preprocessed data loaded from {input_path}")
        return processed_docs

def main():
    """Main preprocessing pipeline."""
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    # Load data
    print("Loading Amazon Fine Foods Reviews dataset...")
    df = preprocessor.load_data('data/finefoods.txt')
    print(f"Loaded {len(df)} reviews")
    
    # Sample data for faster processing (remove this line for full dataset)
    df_sample = df.sample(n=min(10000, len(df)), random_state=42)
    print(f"Using sample of {len(df_sample)} reviews for preprocessing")
    
    # Preprocess documents
    processed_docs = preprocessor.preprocess_documents(df_sample['combined_text'].tolist())
    
    # Save preprocessed data
    os.makedirs('data/processed', exist_ok=True)
    preprocessor.save_preprocessed_data(processed_docs, 'data/processed/preprocessed_docs.pkl')
    
    # Save sample dataframe
    df_sample.to_pickle('data/processed/sample_data.pkl')
    
    print(f"Preprocessing complete! Processed {len(processed_docs)} documents")
    print(f"Average document length: {np.mean([len(doc) for doc in processed_docs]):.2f} tokens")

if __name__ == "__main__":
    main()
