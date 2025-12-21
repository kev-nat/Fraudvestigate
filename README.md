# Fraudvestigate

### Project Overview
This project implements a hybrid LLM agent system for analyzing credit card fraud data. It combines structured database querying with unstructured knowledge retrieval, controlled by an intelligent router.

1. **Text-to-SQL Chain:** Queries specific transaction records from PostgreSQL (e.g., "How many fraud cases happened in NY?").

2. **RAG Agent:** Answers qualitative questions about fraud methodologies using a "Self-Correcting" retrieval system (e.g., "What are common phishing techniques?").

3. **Master Router:** An intelligent classifier that directs user queries to the correct tool.

The system uses **PostgreSQL** database for structured transaction data and **ChromaDB** for vector storage, orchestrated via a Streamlit UI.

--------

### Setup & Installation
**1. Prerequisites (Ollama & Dependencies)**
This project uses local models for embeddings. You must install Ollama and pull the required models.

   - **Install Ollama**  
      Download from [**Ollama**](https://ollama.com).

   - **Pull Models**  
      Open your terminal and run:
      ```bash
      ollama pull qwen3-embedding:0.6b
      ```

   - **Install Dependencies**  
      ```bash
      pip install -r requirements.txt
      ```

**2. Data Preparation**
- **Source:** Download the Credit Card Fraud Detection dataset from [**Kaggle**](https://www.kaggle.com/datasets/kartik2112/fraud-detection) and business review from Tata Consultancy Services [**Understanding Credit Card Frauds**](https://popcenter.asu.edu/sites/g/files/litvpz3631/files/problems/credit_card_fraud/PDFs/Bhatla.pdf)
- **Preprocessing:**
    * Perform Exploratory Data Analysis (EDA) to check for missing values.
    * Ensure data consistency & types: Verify that dates are actual datetime objects and numerical fields don't contain hidden characters.
    * Merge split datasets into a single master `.csv` file.

**3. Database Setup (PostgreSQL)**
- **Create Schema:** Object Explorer -> Schemas -> Tables -> "Create New Table".
- **Define Columns:** Use the expanded schema specifications below to match the dataset structure.

**Database Schema & Justification:** (*Based on the actual database configuration*)

| Column Name         | Data Type                  | Length/Precision | Scale | Notes                                                                 |
|---------------------|----------------------------|------------------|-------|-----------------------------------------------------------------------|
| trans_date_trans_time | TIMESTAMP WITHOUT TIME ZONE | -                | -     | Stores exact date and time for temporal queries.                      |
| credit_card_number  | VARCHAR                   | -                | -     | Identifier, not numeric. Preserves leading zeros.                     |
| merchant            | VARCHAR                   | -                | -     | Variable-length text for merchant names.                              |
| category            | VARCHAR                   | -                | -     | Categorical strings (e.g., “grocery_pos”, “gas_transport”).           |
| transaction_amount  | NUMERIC                   | 10               | 2     | Exact precision for currency values.                                  |
| first               | VARCHAR                   | -                | -     | First name as text.                                                   |
| last                | VARCHAR                   | -                | -     | Last name as text.                                                    |
| gender              | VARCHAR                   | -                | -     | Short text string (M/F).                                              |
| street              | VARCHAR                   | -                | -     | Street address.                                                       |
| city                | VARCHAR                   | -                | -     | City name.                                                            |
| state               | VARCHAR                   | -                | -     | State name.                                                           |
| zip                 | VARCHAR                   | -                | -     | Zip codes can start with zero, so stored as text.                     |
| lat                 | NUMERIC                   | 9                | 6     | Latitude with high precision.                                         |
| long                | NUMERIC                   | 9                | 6     | Longitude with high precision.                                        |
| city_pop            | INTEGER                   | -                | -     | Population counts are whole numbers.                                  |
| job                 | VARCHAR                   | -                | -     | Job titles as text.                                                   |
| dob                 | DATE                      | -                | -     | Date of birth (no time component).                                    |
| transaction_id      | VARCHAR                   | -                | -     | Unique alphanumeric identifier.                                       |
| unix_time           | BIGINT                    | -                | -     | Large integer for Unix timestamps.                                    |
| merchant_latitude   | NUMERIC                   | 9                | 6     | Merchant latitude with precision.                                     |
| merchant_longitude  | NUMERIC                   | 9                | 6     | Merchant longitude with precision.                                    |
| is_fraud            | INTEGER                   | -                | -     | Binary flag (0 = Legitimate, 1 = Fraud).                              |
| dataset_source      | VARCHAR                   | -                | -     | Tracks origin of the record for auditing.                             |

**Notes:**
- Monetary values use `NUMERIC(10,2)` to avoid floating-point precision issues.
- Credit card numbers and ZIP codes are stored as `VARCHAR` to preserve leading zeros.
- Geospatial fields use 6 decimal places for accurate mapping.

- **Import Data:**
    - Newly created table -> Import Data.
    - Select the cleaned `.csv` file.
    - **Encoding:** Select `UTF8` (Handling special characters in merchant names).
    - **Header:** Toggle "Yes" (Skip the first row of column names).

**4. Tool Definition (The "Tools")**
- Convert the **SQL Chain (Part 1)** and RAG Agent **(Part 2)** into functional "Tools".
- These are exposed to the **Master Router**, allowing the system to treat them as distinct capabilities.

--------

### Part 1: LLM SQL Chain
This module enables the system to interact with PostgreSQL database using natural language.

**Workflow:** `Question` -> `SQL Generation` -> `Execution` -> `Natural Language Response`

Key Differentiator:
- **Dynamic Schema Injection:** Unlike basic chains, this pipeline dynamically pulls the current database schema (table names, column types) and injects it into the prompt. This ensures the LLM generates hallucination-free SQL that strictly adheres to the actual database structure.
- **Evaluation**: The notebook includes an evaluation loop that compares the generated SQL against a set of "Gold Standard" queries to ensure accuracy.

### Part 2: RAG Agent
This module handles unstructured queries (knowledge base) using an Active RAG architecture.

**How it differs from Traditional RAG:** A standard RAG pipeline simply retrieves documents and forces an answer. This agent is "Self-Reflective":

1. **Retrieval:** Fetches documents from ChromaDB based on the user query.

2. **Grading (The "Judge"):** The agent evaluates the retrieved documents for relevance.
   - *Is this document actually related to the question?*

3. **Conditional Logic:**
   - **If Relevant:** It proceeds to generate an answer.
   - **If Irrelevant:** It triggers a **Query Transformation**. The agent rewrites the user's search query to be more effective and tries retrieving again rather than hallucinating an answer.
   - **Final Generation:** Synthesizes the answer only using verified, high-quality context.

### Part 3: Master Router
This is the "Traffic Controller" of the system.

- **Function:** It analyzes user's input intent and routes it to the correct tool.
    * User asks for specific data? -> Route to SQL Tool.
    * User asks for general info? -> Route to RAG Tool.

- **Evaluation:** The routing logic and performance are evaluated in Notebook 04, which tests the classifier's accuracy in distinguishing between database queries and general questions.


--------

### Caching & Configuration

**Building the Vector Database** (`build_db.py`): Before running the app, run this script once to ingest the knowledge base. It creates embeddings and persists them to a local `chroma_db` folder, acting as a cache to prevent re-indexing on every restart.


**Configuration** (`config.yaml`): All system parameters are centralized here for easy debugging and model swapping.

- **Model:**
    * LLM: `gpt-4o-mini`
    * Embeddings: `qwen3-embedding:0.6b`
- **Database Connection:** Host, port, and credentials
- **Document Paths:** Relative paths to the knowledge base text files/PDFs 
- **Prompts:** System prompts for the SQL generator, the RAG grader, and the Master Router

--------

### Running the Application (`app.py`)
1. Run `python build_db.py` (**First time only**)
2. Run `streamlit run app.py`
