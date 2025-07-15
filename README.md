# SDG Analysis Scripts

Python scripts for analyzing United Nations Sustainable Development Goals (SDG) progress reports. These tools support content search and evaluation for Copilot agents created in Copilot Studio, developed during an internship collaboration between the United Nations and Microsoft.

## Overview

This repository contains two analysis scripts designed to process Voluntary National Review (VNR) reports and evaluate content relevance to SDG indicators. The scripts help improve Copilot agent functionality by creating thematic lexicons and providing relevance scoring for content search capabilities.

## Scripts

### 1. Relevance Evaluation (`RelevanceEvaluation.py`)
Evaluates how well extracted insights align with SDG 1 (No Poverty) indicators using semantic similarity analysis.

**Key Functions:**
- Embeds SDG 1 corpus and insights using sentence transformers (`all-mpnet-base-v2`)
- Calculates cosine similarity scores between insights and SDG indicators
- Ranks insights by relevance with configurable thresholds
- Outputs scored content for Copilot agent training

### 2. Topic Modeling (`TopicModeling.py`)
Discovers thematic patterns across VNR report collections to create searchable lexicons for Copilot agents.

**Key Functions:**
- Processes PDF documents from VNR report database
- Implements Latent Dirichlet Allocation (LDA) using Gensim
- Generates 30 thematic topics with associated keywords
- Creates bigram phrases for enhanced content search

## Requirements

```bash
sentence-transformers
gensim
nltk
numpy
pandas
pymupdf
openpyxl
scikit-learn
```

## Installation

```bash
pip install sentence-transformers gensim nltk numpy pandas pymupdf openpyxl scikit-learn
```

## Usage

### Relevance Evaluation
```bash
python RelevanceEvaluation.py
```
**Output**: Ranked insights with relevance scores, identifying content most aligned with SDG 1 indicators.

### Topic Modeling
```bash
python TopicModeling.py
```
**Output**: 30 thematic topics with top terms and coherence scores for lexicon creation.

## Technical Features

### Relevance Evaluation
- Uses transformer-based embeddings for semantic text analysis
- Provides objective relevance scores (0-1 scale)
- Highlights low-relevance content below configurable thresholds
- Calculates average relevance across document collections

### Topic Modeling
- Preprocesses documents by removing country-specific terms and proper nouns
- Implements advanced tokenization with part-of-speech filtering
- Uses document frequency boosting for improved topic quality
- Includes coherence scoring for topic validation

## Data Processing

### Input Data
- **VNR Reports**: UN Voluntary National Review PDF documents
- **SDG Indicators**: Official SDG 1 targets and indicators text
- **Extracted Insights**: Processed content focusing on poverty reduction

### Preprocessing Pipeline
- PDF text extraction and cleaning
- Tokenization and stopword removal
- Country/region/currency term filtering
- Bigram phrase detection
- POS tagging and proper noun removal

## Applications for Copilot Agents

### Content Search Enhancement
- **Thematic Lexicons**: Topic modeling results provide keyword sets for improved agent search capabilities
- **Relevance Filtering**: Similarity scores help agents prioritize most relevant content
- **Quality Assessment**: Coherence metrics validate topic usefulness for agent training

### Agent Performance Evaluation
- **Benchmark Scoring**: Relevance evaluation provides objective measures for agent output quality
- **Content Validation**: Similarity thresholds help identify when agent responses align with UN objectives
- **Training Data**: Ranked insights serve as high-quality training examples for agent improvement

## Sample Output

### Relevance Scores
```
Score: 0.8234
Insight: Poverty rates increased from 44% to 51.1% between 2016-2020...

Score: 0.7891
Insight: Access to electricity improved from 10% to 24% over 2013-2020...

Score: 0.5432 <<< LOW RELEVANCE
Insight: Gender-responsive budgeting allocated $380,000 in FY 2022...
```

### Topic Examples
```
Topic #1: poverty, reduction, income, household, population, economic
Topic #2: nutrition, health, services, facilities, children, program
Topic #3: water, sanitation, access, infrastructure, rural, urban
```

## Configuration

### Relevance Evaluation Settings
- **Similarity threshold**: Default 0.6 (adjustable for different quality standards)
- **Embedding model**: `all-mpnet-base-v2` (can be changed for different performance characteristics)

### Topic Modeling Parameters
- **Number of topics**: 30 (optimized for VNR corpus size)
- **Passes**: 50 training iterations
- **Minimum word frequency**: 5 occurrences across corpus

## File Structure

```
├── RelevanceEvaluation.py    # Similarity scoring script
├── TopicModeling.py          # LDA topic discovery script
└── README.md                 # This file
```

## Project Context

These scripts were developed during a collaborative internship at the United Nations in New York, focusing on improving Copilot agent capabilities for SDG report analysis. The tools help automate content search and evaluation processes that support UN monitoring of global development progress.

## Notes

- Topic modeling requires local access to VNR PDF collection (path configured in script)
- Relevance evaluation uses pre-extracted insights from Liberia's VNR as sample data
- Scripts are designed to process English-language documents
- Processing time varies based on corpus size and available computational resources
