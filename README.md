# Multimodal Real Estate Search

## Overview

This project is a part of Udacity Generative AI & enables a sophisticated real estate search functionality using  text  inputs. The core of the project combines the strengths of several advanced Gen AI models to deliver accurate and relevant real estate listings based on user queries.

### Key Components

### 1. Generating Real Estate Listings

- **Model Used**: OpenAI's GPT-3.5-turbo 
- **Functionality**: Generates detailed descriptions of real estate listings given a city and country.

### 2. Storing Embeddings in Chroma

- **Vector Store**: Chroma.
- **Functionality**: Stores tesxt embedding for efficient retrieval.

### 3. Generating Output
- **LLM Chain**: RetrievalQA
- **Functionality**: LLM Chain to retrieve relevant listings

