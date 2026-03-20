# **Global Code Generation Rules (Mandatory)**

From this point onward, **all generated code, scripts, and Python programs must strictly follow the rules below**. These rules apply repository-wide and override any default assumptions.

**Your Goal:** Generate **debug-ready, production-quality code** that is **highly readable** and **easy to maintain**.

---

## **1. Best Practices & Anti-Over-Engineering**

### **Core Philosophy**

- **Best Practices:** You must adhere to industry-standard best practices for design, security, and performance.
- **Avoid Over-Engineering:** **Do not over-complicate solutions.** Use the simplest effective approach. Avoid unnecessary abstraction layers, complex class hierarchies, or obscure design patterns unless they provide a clear, necessary benefit.
- **KISS Principle:** Keep It Simple, Stupid.

### **Code Structure**

- **Explicit > Implicit:** Logic must be transparent.
- **Linear Flow:** Prefer straightforward, procedural logic where possible.
- **Modularity:** Break code into small, single-purpose functions with clear inputs and outputs.
- **Type Hinting:** Mandatory for all functions (e.g., `def process(data: dict) -> list:`).

---

## **2. Python Environment & Dependency Management (`uv`)**

### **Virtual Environment**

- **Tool:** Use **`uv`** for Python virtual environment management.
- **Execution:** Always assume the environment is managed by `uv`. Use `uv run <script.py>` or ensure the environment is activated.

### **Dependency Management (`pyproject.toml`)**

- **Single Source of Truth:** `pyproject.toml` is the authority for dependencies.
- **Add Dependencies:** If a new library is required, **you must explicitly handle adding it** to `pyproject.toml`.
- **Remove Unused:** If functionality is removed, **you must remove unused dependencies** from `pyproject.toml`.
- **Cleanliness:** Keep the dependency list minimal and accurate.

---

## **3. Mandatory Logging Standard — `structlog`**

### **No `print()` Usage**

- **Strict Prohibition:** Do NOT use `print()` for any program output.
- **Structured Logging:** Use `structlog` for all logging. Logs must be **structured, event-based, and machine-readable**.

### **Configuration**

- **Standalone:** Include the configuration block at the top of single scripts.
- **Project:** Use a central `utils/logger.py` for configuration.

**Configuration Snippet:**

```python
import structlog
import logging
import sys

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="%H:%M:%S", utc=False),
        structlog.dev.ConsoleRenderer(colors=True),
    ],
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)
logger = structlog.get_logger()
```

### **Data Profiling**

- **Do Not Print Data:** When inspecting data (DataFrames, JSON, etc.), log a profile (shape, columns, types) instead of printing the raw data.

---

## **4. Documentation & Repository Organization**

### **Concise & Centralized**

- **Concise:** Documentation must be clear and to the point. **Avoid garbage information.**
- **No Repetition:** **Avoid repetition at all costs.** Do not duplicate information across multiple files.
- **Single Source:** Keep all related documentation in a **single Markdown file** whenever possible. Do not fracture documentation into many tiny files.

### **Well-Organized Repository**

- **Structure:** Keep the repository clean. Files should be logically organized.
- **Self-Narrating Code:**
  - Use **Google-Style Docstrings** for all modules, classes, and functions.
  - Provide **line-by-line comments** for complex algorithmic logic.
  - Explain the "Why" (business logic) in comments, not just the "How".

---

## **5. Database & CLI Operations**

### **MongoDB Access**

- **Connection:** The `MONGODB_URI` is located in the `.env` file at the repository root.
- **CLI Tool:** Always use `mongosh` to interact with the database from the command line.
  - *Login command:* `mongosh $(grep MONGODB_URI .env | cut -d '=' -f2)` (or extract the URI manually from `.env`).
- **Basic Commands:**
  - Check databases: `show dbs`
  - Switch database: `use <database_name>`
  - Check collections: `show collections`
  - Query example: `db.<collection_name>.find().limit(5)`

### **Useful MongoDB Aggregation Queries**

When inspecting the generated Q&A datasets (`wot_rag_qna` or `wot_qna`), use these aggregation queries to check generation statistics:

**1. Count total documents grouped by the AI model used:**
```javascript
db.wot_rag_qna.aggregate([
  { $group: { _id: "$metadata.generation_model", total_documents: { $sum: 1 } } },
  { $sort: { total_documents: -1 } }
])
```

**2. Count documents grouped by both AI model AND category:**
```javascript
db.wot_qna.aggregate([
  { 
    $group: { 
      _id: { model: "$metadata.generation_model", category: "$category" }, 
      total_documents: { $sum: 1 } 
    } 
  },
  { $sort: { "_id.model": 1, total_documents: -1 } }
])
```

---

## **Enforcement**

1.  **Refactor immediately** if a rule is violated.
2.  **No permission needed** to enforce these standards.
3.  **Mandatory:** These rules are non-negotiable.
