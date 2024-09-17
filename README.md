# RagChat

RagChat is a chatbot application that combines a Large Language Model (LLM) with Retrieval-Augmented Generation (RAG) to answer questions based on custom datasets. It allows you to integrate a local LLM with a retrieval mechanism to provide context-aware, accurate answers.

## Prerequisites

Before you get started, ensure you have the following installed on your machine:

- [Ollama](https://ollama.com) (for running LLMs locally)
- Python 3.8 or higher

## Setup and installation

Follow these steps to get started with RagChat:

### 1. Install Ollama
Download and install [Ollama](https://ollama.com), which is required to run models locally.

### 2. Pull LLaMA 3 (or another model)
Once Ollama is installed, pull the LLaMA 3 model (or another model you prefer):
```bash
ollama pull llama3
```

### 3. Clone this repo
```bash
git clone https://github.com/grippvh/ragChat.git
cd ragChat
```

### 4. Install required libraries
```bash
pip install -r requirements.txt
```
## Run the application
```bash
./run.sh
```
