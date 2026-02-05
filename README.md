## 1. Core Components

### A. Web Chat Interface

**Purpose:**
Primary interaction layer where farmers communicate with the chatbot.

**Responsibilities:**

* Accept user text queries
* Accept image uploads for disease diagnosis
* Display chatbot responses in conversational format
* Maintain session level conversation context

### B. Backend API Layer

**Purpose:**
Acts as the central entry point for all user interactions and system orchestration.

**Responsibilities:**

* Receive requests from the web chat
* Manage session state and conversation history
* Route user queries to the agent orchestration layer
* Return final responses to UI
* Handle logging and error handling

### C. Agent Orchestrator (Conversation Brain)

**Purpose:**
Determines user intent and dynamically selects the correct tool or capability.

**Responsibilities:**

* Understand user queries using LLM reasoning
* Decide when to call disease detection, mandi price retrieval, or scheme retrieval
* Combine tool outputs into farmer-friendly responses
* Maintain conversation continuity across multiple turns

### D. Disease Detection Module

**Purpose:**
Identify crop diseases from images and provide treatment recommendations.

**Responsibilities:**

* Accept farmer uploaded crop images
* Use multimodal LLMs to detect diseases
* Generate treatment plans including pesticide suggestions
* Provide preventive care recommendations
* Add safety disclaimers and uncertainty handling

### E. Mandi Price Retrieval Module

**Purpose:**
Provide real-time region-specific crop price information.

**Responsibilities:**

* Extract crop and location details from user query
* Fetch mandi prices from external agricultural market APIs
* Format and present price insights in simple language
* Handle API failures or missing data gracefully

### F. Government Scheme Retrieval Module (RAG)

**Purpose:**
Help farmers understand agricultural government schemes.

**Responsibilities:**

* Process and index scheme PDF documents
* Retrieve relevant scheme information based on user questions
* Summarize eligibility, benefits, and application steps
* Provide simplified explanations suitable for farmers

### G. Evaluation and Observability Layer

**Purpose:**
Measure response quality and monitor system behavior.

**Responsibilities:**

* Evaluate LLM outputs using automated evaluation frameworks
* Track tool selection accuracy
* Measure response correctness and factual grounding
* Log conversations, tool calls, and latency metrics

## 2. Tech Stack

### Programming Language

* Python

### Dependency Management

* uv

### Backend Framework

* FastAPI

### LLM and Multimodal Processing

* OpenAI multimodal models (for text + image understanding)
* OpenAI embeddings (for retrieval tasks)

### Retrieval Augmented Generation

* Qdrant for vector database and similarity search
* PDF text extraction using PyMuPDF or equivalent libraries

### External Data Integration

* Indian agriculture market APIs (e.g., Agmarknet / data.gov.in datasets)

### Evaluation Framework

* DeepEval

### Web Interface

* Streamlit (for rapid MVP development)

### Logging and Monitoring

* Using Python loguru package

### Deployment

* Lightweight cloud hosting platforms for Render
