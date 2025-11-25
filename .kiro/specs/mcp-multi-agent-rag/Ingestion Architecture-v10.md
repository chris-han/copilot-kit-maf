Here is the **Final Architecture Design Document (Version 11.0)**.

This version incorporates the **Canonical Schema Transformation logic**, the **MDM/Ontology strategy**, and the **Updated Topology** with the ClickHouse loading branch prioritized to the left.

---

# Architecture Design: Intelligent Ingestion, MDM & Discovery (v11.0)

**System:** Enterprise Universal Ingestion Pipeline
**Core Philosophy:** "Bifurcated Processing, Unified Discovery, Semantic Integration."
**Architecture Pattern:** The "Tri-Store" (ClickHouse + Milvus + Postgres).

## 1. Executive Summary

This system solves three distinct problems in Enterprise RAG:
1.  **Analytical Scale:** Storing billions of raw rows for fast SQL (ClickHouse).
2.  **Semantic Discovery:** Finding the right data table using natural language (Milvus + BGE-M3).
3.  **Data Integration (MDM):** Mapping diverse source schemas to a Canonical Ontology using GenAI (Postgres + Polars).

The pipeline uses **Polars** for high-performance compute and **Llama-3.1** (Green Nodes) for reasoning, avoiding "dumb" keyword matching for schema alignment.

---

## 2. System Topology

**Change Note:** The "Raw Loading" path (ClickHouse) is moved to the far left to indicate it is the foundational step, running in parallel with intelligence tasks.

```mermaid
graph TD
    %% Styling
    classDef llm fill:#A5D6A7,stroke:#2E7D32,stroke-width:2px,color:black;
    classDef storage fill:#E1F5FE,stroke:#0277BD,stroke-width:2px,color:black;
    classDef logic fill:#FFFFFF,stroke:#333,stroke-width:1px;

    subgraph "Input Layer"
        File["Input File<br/>(PDF, CSV, Parquet)"]
    end

    subgraph "Ingestion Worker (Python/Polars)"
        direction TB
        Router{<B>Router</B>}:::logic

        %% --- Path A: Unstructured ---
        subgraph "Path A: Docs"
            Chunker["<B>Chunking</B>"]:::logic
            NER_Doc["<B>GLiNER</B><br/>(Extract Entities)"]:::logic
            EmbedDoc["<B>BGE-M3</B><br/>(Embed Text)"]:::logic
        end

        %% --- Path B: Structured ---
        subgraph "Path B: Data"
            Profiler["<B>Polars Profiler</B><br/>(High-Speed Scan)<br/>Extracts: Cardinality, Precision, Samples"]:::logic
            
            %% Branch 2: Value Extraction
            NER_Val["<B>GLiNER (Value Scan)</B><br/>Target: Text Cells"]:::logic
            
            %% Branch 3: Schema Mapping (MDM)
            MapGen["<B>LLM Schema Mapper</B><br/>Input: Headers + Samples<br/>Output: JSON Map"]:::llm
            
            Loader["<B>ClickHouse Loader</B><br/>(Stream Raw Rows)"]:::logic
            Transformer["<B>Canonical Transformer</B><br/>(Polars Execution)<br/>Apply JSON Map to Data"]:::logic
            
            %% Branch 4: Semantic Discovery
            DescGen["<B>LLM Summarizer</B><br/>Input: Stats<br/>Output: NL Description"]:::llm
            EmbedSchema["<B>BGE-M3</B><br/>(Embed Summary)"]:::logic
        end
    end

    subgraph "The Tri-Store Storage Layer"
        %% 1. Analytics Data
        CH[("<B>ClickHouse</B><br/>(Raw & Canonical Data)")]:::storage

        %% 2. Metadata / Ontology
        PG[("<B>PostgreSQL</B><br/>(Control Plane)<br/>1. Master Ontology<br/>2. Column Catalog<br/>3. Schema Maps")]:::storage
        
        %% 3. Vector Indices
        Milvus[("<B>Milvus</B><br/>(Search Engine)<br/>Coll: docs, data_catalog")]:::storage
    end

    %% Routing
    File --> Router
    Router -->|"PDF/Doc"| Chunker
    Router -->|"CSV/Log"| Profiler

    %% Path A Flow
    Chunker --> NER_Doc
    NER_Doc -->|"New Entities"| PG
    NER_Doc --> EmbedDoc -->|"Doc Vector"| Milvus

    %% Path B Flow
    Profiler -->|"1. Raw Rows"| Loader --> CH
    Profiler -->|"2. Text Samples"| NER_Val -->|"New Entities"| PG
    
    
    %% MDM Mapping Flow
    Profiler -->|"3. Schema + Samples"| MapGen
    MapGen -->|"JSON Map"| PG
    MapGen -->|"JSON Map"| Transformer
    Transformer -->|"Canonical Rows"| CH
    
    %% Discovery Flow
    Profiler -->|"4. Schema Stats"| DescGen
    DescGen -->|"Text Description"| EmbedSchema -->|"Vector"| Milvus

```

---

## 3. Storage Schema Strategy (The Tri-Store)

### A. ClickHouse (Analytical Engine)
*   **Raw Tables:** `raw_sales_2024` (As ingested).
*   **Canonical Tables:** `std_sales_transactions` (Transformed via mapped schema).

### B. PostgreSQL (Control Plane)
*   **`column_catalog`:** Physical stats (Precision, Scale, Cardinality).
*   **`ontology_master`:** Entities found by GLiNER (e.g., "Project Alpha").
*   **`schema_mappings`:** The JSON logic linking Raw -> Canonical.

### C. Milvus (Discovery Engine)
*   **`data_catalog`:** Embeddings of *table descriptions*. Allows Agent to find tables by asking "Show me revenue" even if the column is named `t_val`.

---

## 4. Path B: Structured Data Pipeline Logic

### Step 1: The Smart Profiler (Polars)
Extracts semantic roles and precision without performance-heavy `describe()`.

```python
import polars as pl

def profile_columns(df: pl.DataFrame):
    profiles = []
    for col in df.columns:
        dtype = df[col].dtype
        
        # 1. Decimal Precision (Critical for FinOps)
        precision, scale = None, None
        if isinstance(dtype, pl.Decimal):
            precision = dtype.precision
            scale = dtype.scale
            
        # 2. Semantic Role Heuristic
        n_unique = df[col].n_unique()
        role = "DIMENSION"
        if dtype in [pl.Float64, pl.Float32, pl.Decimal] and n_unique > 50:
            role = "METRIC"
        elif dtype in [pl.Date, pl.Datetime]:
            role = "TIME_INDEX"
            
        profiles.append({
            "name": col,
            "native_type": str(dtype),
            "role": role,
            "precision": precision,
            "scale": scale,
            "samples": df[col].drop_nulls().head(3).to_list()
        })
    return profiles
```

### Step 2: The LLM Schema Mapper (Green Node)
Maps cryptic source columns to your **Canonical Ontology**.

*   **Input:** `{"name": "val_amt", "samples": ["100.50"]}`
*   **Prompt:** "Map this to Standard Finance Ontology (Transaction.Amount, Customer.ID)."
*   **Output (JSON):** `{"val_amt": "Transaction.Amount"}`

### Step 3: Canonical Transformation (Polars)
This utilizes the JSON output from Step 2 to physically transform the data using Polars expressions.

```python
class CanonicalTransformer:
    def __init__(self, mapping_json: dict, canonical_types: dict):
        self.mapping = {k: v for k, v in mapping_json.items() if v is not None}
        self.canonical_types = canonical_types

    def execute(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Transforms Raw DF -> Canonical DF in one lazy execution.
        """
        expressions = []
        for source_col, target_col in self.mapping.items():
            if source_col not in df.columns: continue
            
            # 1. Get Target Type (e.g., pl.Float64)
            target_dtype = self.canonical_types.get(target_col, pl.Utf8)
            
            # 2. Build Expression: Rename + Cast
            # strict=False turns bad data into Nulls (Safe ETL)
            expr = pl.col(source_col).cast(target_dtype, strict=False).alias(target_col)
            expressions.append(expr)
            
        # 3. Select (Filters out unmapped columns automatically)
        return df.select(expressions)
```

### Step 4: Semantic Discovery (Green Node)
Generates the text that goes into **Milvus**.

*   **Prompt:** "Describe this table based on these metrics and dimensions."
*   **Output:** "This table contains financial transactions for 2024, tracking Metrics like 'Transaction.Amount' across Dimensions like 'Region'."
*   **Action:** Embed via **BGE-M3** -> Milvus `data_catalog`.

---

## 5. Path A: Unstructured Pipeline Logic

1.  **Chunking:** `unstructured.partition`.
2.  **Enrichment (GLiNER):**
    *   Extracts `Project`, `Competitor`, `Vendor`.
    *   **Action:** Upsert to Postgres `ontology_master`.
    *   *Benefit:* The Orchestrator now knows these are valid entities when querying Structured Data later.
3.  **Vectorization:** BGE-M3 (Dense + Sparse) -> Milvus `docs`.

---

## 6. Infrastructure Stack (Docker)

```yaml
version: '3.8'
services:
  # 1. Ingestion Worker (The Brain)
  ingestion:
    image: python:3.11-slim
    environment:
      - CLICKHOUSE_URI=clickhouse://default:password@clickhouse:8123
      - MILVUS_URI=http://milvus:19530
      - PG_URI=postgresql://user:pass@postgres:5432/metadata
      - OLLAMA_HOST=http://ollama:11434
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu] # For BGE-M3 & GLiNER

  # 2. Reasoning Engine
  ollama:
    image: ollama/ollama:latest
    volumes:
      - ollama_data:/root/.ollama
    # Model: llama3.1:8b

  # 3. The Tri-Store
  clickhouse:
    image: clickhouse/clickhouse-server:latest
    ports: ["8123:8123"]
    
  milvus:
    image: milvusdb/milvus:v2.3.4
    ports: ["19530:19530"]
    
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: metadata
```