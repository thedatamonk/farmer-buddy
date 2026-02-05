# Project Kisan - Testing Guide

## Prerequisites

### Required Services
1. **Python 3.11+** - The application requires Python 3.11 or later
2. **Qdrant Vector Database** - Required for the government schemes RAG feature
3. **OpenAI API Key** - Required for LLM features (chat, vision, embeddings)

### Optional Services
- **Data.gov.in API Key** - For live mandi price data (works without, but returns no data)

## Setup Instructions

### 1. Install Dependencies

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
pip install -e ".[dev]"  # For development dependencies
```

### 2. Configure Environment

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your API keys:
# - OPENAI_API_KEY (required)
# - MANDI_API_KEY (optional, for live mandi prices)
```

### 3. Start Qdrant (for RAG features)

```bash
# Using Docker
docker run -d -p 6333:6333 qdrant/qdrant

# Verify Qdrant is running
curl http://localhost:6333/collections
```

### 4. Index Government Scheme Documents (Optional)

```bash
# Add PDF documents to data/schemes/ directory, then run:
PYTHONPATH=src python scripts/index_schemes.py
```

### 5. Start the API Server

```bash
# Development mode with auto-reload
PYTHONPATH=src uvicorn kisan.api.main:app --reload --port 8080

# Production mode
PYTHONPATH=src uvicorn kisan.api.main:app --host 0.0.0.0 --port 8080
```

### 6. Start the Streamlit UI (Optional)

```bash
# In a separate terminal
cd ui
streamlit run app.py
```

## Running Tests

### Unit Tests

```bash
# Run all unit tests
PYTHONPATH=src pytest tests/unit -v

# Run specific test file
PYTHONPATH=src pytest tests/unit/test_session.py -v

# Run with coverage
PYTHONPATH=src pytest tests/unit --cov=kisan --cov-report=html
```

### Integration Tests

```bash
# Run integration tests (doesn't require external services)
PYTHONPATH=src pytest tests/integration -v
```

### All Tests

```bash
# Run the full test suite
PYTHONPATH=src pytest tests/ -v
```

### Evaluation Tests (Optional)

```bash
# Install deepeval first
pip install deepeval

# Run evaluation tests
PYTHONPATH=src pytest tests/evaluation -v
```

## Manual Testing

### 1. Health Check

```bash
curl http://localhost:8080/health
```

Expected response:
```json
{"status": "healthy", "service": "kisan", "version": "0.1.0"}
```

### 2. Basic Chat (No Tools)

```bash
curl -X POST http://localhost:8080/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, how are you?"}'
```

### 3. Government Scheme Query (RAG)

```bash
curl -X POST http://localhost:8080/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is PM-KISAN scheme?"}'
```

### 4. Mandi Price Query

```bash
curl -X POST http://localhost:8080/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the price of wheat in Delhi?"}'
```

### 5. Disease Detection (with Image)

```bash
# First, encode an image to base64
IMAGE_BASE64=$(base64 -i path/to/crop_image.jpg)

# Then send the request
curl -X POST http://localhost:8080/api/v1/chat \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"What disease does this crop have?\", \"image\": \"$IMAGE_BASE64\"}"
```

### 6. Session Continuity

```bash
# First message - note the session_id in response
curl -X POST http://localhost:8080/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "I grow wheat in Punjab"}'

# Follow-up message with session_id
curl -X POST http://localhost:8080/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What diseases should I watch for?", "session_id": "<session_id_from_above>"}'
```

### 7. Get Session History

```bash
curl http://localhost:8080/api/v1/sessions/<session_id>
```

### 8. Delete Session

```bash
curl -X DELETE http://localhost:8080/api/v1/sessions/<session_id>
```

## Test Scenarios

### Feature: Chat Functionality
| Test Case | Input | Expected Behavior |
|-----------|-------|-------------------|
| Basic greeting | "Hello" | Friendly greeting response |
| Hindi message | "नमस्ते" | Response in Hindi |
| Hinglish message | "Meri fasal mein problem hai" | Response in Hinglish |
| Empty message | "" | 422 validation error |
| Long message (>4000 chars) | Very long text | 422 validation error |

### Feature: Disease Detection
| Test Case | Input | Expected Behavior |
|-----------|-------|-------------------|
| Valid crop image | Image + "What's wrong with my crop?" | Disease analysis with treatments |
| No image provided | "Analyze my crop" (no image) | Request for image upload |
| Non-crop image | Random image | Polite message asking for crop photo |
| Low quality image | Blurry image | Analysis with lower confidence |

### Feature: Mandi Prices
| Test Case | Input | Expected Behavior |
|-----------|-------|-------------------|
| Valid commodity | "wheat price" | Price data from multiple markets |
| With location | "onion price in Maharashtra" | Filtered price data |
| Invalid commodity | "xyz123 price" | "No data found" message |
| Hindi query | "गेहूं का भाव" | Price data (understands Hindi) |

### Feature: Government Schemes
| Test Case | Input | Expected Behavior |
|-----------|-------|-------------------|
| Specific scheme | "What is PM-KISAN?" | Detailed scheme information |
| Eligibility query | "Am I eligible for crop insurance?" | Eligibility criteria |
| Application process | "How to apply for KCC?" | Step-by-step application guide |
| Unknown scheme | "Tell me about XYZ scheme" | "Couldn't find information" |

### Feature: Session Management
| Test Case | Input | Expected Behavior |
|-----------|-------|-------------------|
| New session | First message without session_id | New session created |
| Continue session | Message with valid session_id | Context preserved |
| Invalid session | Message with invalid session_id | New session created |
| Get session | GET /sessions/{id} | Full conversation history |
| Delete session | DELETE /sessions/{id} | Session removed |

## Performance Testing

### Load Testing with wrk

```bash
# Install wrk
brew install wrk  # macOS

# Run load test on health endpoint
wrk -t4 -c100 -d30s http://localhost:8080/health

# Run load test on chat endpoint
wrk -t4 -c10 -d30s -s post.lua http://localhost:8080/api/v1/chat
```

Create `post.lua`:
```lua
wrk.method = "POST"
wrk.headers["Content-Type"] = "application/json"
wrk.body = '{"message": "Hello"}'
```

## Linting and Code Quality

```bash
# Run ruff linter
uv run ruff check src/ tests/

# Auto-fix issues
uv run ruff check src/ tests/ --fix

# Format code
uv run ruff format src/ tests/
```

## Troubleshooting

### Common Issues

1. **"Connection refused" on Qdrant**
   - Ensure Qdrant is running: `docker ps`
   - Check the port: `curl http://localhost:6333`

2. **"Invalid API key" from OpenAI**
   - Verify OPENAI_API_KEY in .env file
   - Check key is valid: `echo $OPENAI_API_KEY`

3. **"No module named 'kisan'"**
   - Set PYTHONPATH: `export PYTHONPATH=src`
   - Or run with: `PYTHONPATH=src uvicorn ...`

4. **"No price data found"**
   - Mandi API requires API key from data.gov.in
   - Without key, returns empty results (not an error)

5. **Tests failing with import errors**
   - Run: `uv sync` to install all dependencies
   - Ensure Python 3.11+: `python --version`

### Checking Logs

```bash
# Server logs are printed to stderr by default
# For file logs, check logs/ directory (created in production mode)

# Enable debug logging
LOG_LEVEL=DEBUG PYTHONPATH=src uvicorn kisan.api.main:app --port 8080
```

## API Documentation

Once the server is running, access:
- **Swagger UI**: http://localhost:8080/docs
- **ReDoc**: http://localhost:8080/redoc
- **OpenAPI JSON**: http://localhost:8080/openapi.json
