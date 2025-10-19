#!/usr/bin/env python3
"""
Test script to verify the topic modeling pipeline setup.
Run this script to test if all dependencies are properly installed.
"""

import sys
import os

def test_imports():
    """Test if all required packages can be imported."""
    print("Testing package imports...")
    
    try:
        import pandas as pd
        print("✓ pandas imported successfully")
    except ImportError as e:
        print(f"✗ pandas import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✓ numpy imported successfully")
    except ImportError as e:
        print(f"✗ numpy import failed: {e}")
        return False
    
    try:
        import sklearn
        print("✓ scikit-learn imported successfully")
    except ImportError as e:
        print(f"✗ scikit-learn import failed: {e}")
        return False
    
    try:
        import nltk
        print("✓ nltk imported successfully")
    except ImportError as e:
        print(f"✗ nltk import failed: {e}")
        return False
    
    try:
        import gensim
        print("✓ gensim imported successfully")
    except ImportError as e:
        print(f"✗ gensim import failed: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✓ matplotlib imported successfully")
    except ImportError as e:
        print(f"✗ matplotlib import failed: {e}")
        return False
    
    try:
        import seaborn as sns
        print("✓ seaborn imported successfully")
    except ImportError as e:
        print(f"✗ seaborn import failed: {e}")
        return False
    
    try:
        import plotly
        print("✓ plotly imported successfully")
    except ImportError as e:
        print(f"✗ plotly import failed: {e}")
        return False
    
    try:
        import wordcloud
        print("✓ wordcloud imported successfully")
    except ImportError as e:
        print(f"✗ wordcloud import failed: {e}")
        return False
    
    try:
        import faiss
        print("✓ faiss imported successfully")
    except ImportError as e:
        print(f"✗ faiss import failed: {e}")
        return False
    
    try:
        import pyLDAvis
        print("✓ pyLDAvis imported successfully")
    except ImportError as e:
        print(f"✗ pyLDAvis import failed: {e}")
        return False
    
    try:
        import sentence_transformers
        print("✓ sentence-transformers imported successfully")
    except ImportError as e:
        print(f"✗ sentence-transformers import failed: {e}")
        return False
    
    return True

def test_nltk_data():
    """Test if required NLTK data is available."""
    print("\nTesting NLTK data...")
    
    try:
        import nltk
        
        # Test punkt tokenizer
        try:
            nltk.data.find('tokenizers/punkt')
            print("✓ NLTK punkt tokenizer available")
        except LookupError:
            print("✗ NLTK punkt tokenizer not found - will download automatically")
        
        # Test stopwords
        try:
            nltk.data.find('corpora/stopwords')
            print("✓ NLTK stopwords available")
        except LookupError:
            print("✗ NLTK stopwords not found - will download automatically")
        
        # Test wordnet
        try:
            nltk.data.find('corpora/wordnet')
            print("✓ NLTK wordnet available")
        except LookupError:
            print("✗ NLTK wordnet not found - will download automatically")
        
        return True
        
    except Exception as e:
        print(f"✗ NLTK data test failed: {e}")
        return False

def test_data_file():
    """Test if the data file exists."""
    print("\nTesting data file...")
    
    data_file = 'data/finefoods.txt'
    if os.path.exists(data_file):
        print(f"✓ Data file found: {data_file}")
        
        # Check file size
        file_size = os.path.getsize(data_file)
        print(f"  File size: {file_size / (1024*1024):.2f} MB")
        
        return True
    else:
        print(f"✗ Data file not found: {data_file}")
        print("  Please ensure the finefoods.txt file is in the data/ directory")
        return False

def test_project_structure():
    """Test if the project structure is correct."""
    print("\nTesting project structure...")
    
    required_dirs = ['src', 'data', 'notebooks', 'results']
    required_files = ['requirements.txt', 'README.md']
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✓ Directory found: {dir_name}/")
        else:
            print(f"✗ Directory missing: {dir_name}/")
            return False
    
    for file_name in required_files:
        if os.path.exists(file_name):
            print(f"✓ File found: {file_name}")
        else:
            print(f"✗ File missing: {file_name}")
            return False
    
    return True

def main():
    """Main test function."""
    print("=" * 60)
    print("TOPIC MODELING PIPELINE - SETUP TEST")
    print("=" * 60)
    
    all_tests_passed = True
    
    # Test project structure
    if not test_project_structure():
        all_tests_passed = False
    
    # Test data file
    if not test_data_file():
        all_tests_passed = False
    
    # Test package imports
    if not test_imports():
        all_tests_passed = False
    
    # Test NLTK data
    if not test_nltk_data():
        all_tests_passed = False
    
    print("\n" + "=" * 60)
    if all_tests_passed:
        print("✓ ALL TESTS PASSED!")
        print("The topic modeling pipeline is ready to use.")
        print("\nNext steps:")
        print("1. Run: python src/main.py")
        print("2. Or explore interactively: jupyter notebook notebooks/topic_modeling_analysis.ipynb")
    else:
        print("✗ SOME TESTS FAILED!")
        print("Please fix the issues above before running the pipeline.")
        print("\nTo install dependencies:")
        print("pip install -r requirements.txt")
        print("python -m spacy download en_core_web_sm")
    print("=" * 60)

if __name__ == "__main__":
    main()
