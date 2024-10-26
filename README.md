# Medical Question-Answering Service

A FastAPI-based service that uses Retrieval-Augmented Generation (RAG) to answer medical questions based on MSD Manual and CBIP data.

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv\Scripts\activate  
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file with:
```
GROQ_API_KEY=your_groq_api_key_here
```

4. Prepare your data:
- Load your csv data into a vector database

## Running the Service

1. Start the server:
```bash
python serv2.py
```

The service will be available at `http://localhost:8000`

## API Usage

Send questions to the `/answer` endpoint:

```bash
curl -X POST "http://localhost:8000/answer" \
     -H "Content-Type: application/json" \
     -d '{"question": "What is the recommended treatment for type 2 diabetes?"}'
```

## Running Tests

Execute the test suite:
```bash
pytest test2.py -v
```

## Features

- Document preprocessing and chunking
- FAISS vector store for efficient similarity search
- Integration with Groq's Llama 3.2 3b model
- Error handling and logging
- FastAPI-based API
- Comprehensive test suite

## Error Handling

The service handles various error cases:
- Empty questions
- Invalid request formats
- Server-side errors
- Valid request formats

## Logging

Logs are written to the console and include:
- Incoming requests
- Error messages
- Service initialization status

## Architecture

1. **Data Processing Pipeline:**
   - Document loading
   - Text preprocessing
   - Vector store indexing

2. **Query Pipeline:**
   - Question reception
   - Relevant document retrieval
   - Answer generation using Groq LLM

3. **API Layer:**
   - FastAPI endpoints
   - Request/response validation
   - Error handling

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request