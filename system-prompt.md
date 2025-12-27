## **Global Code Generation Rules (Mandatory)**

From this point onward, **all generated code, scripts, and Python programs must strictly follow the rules below**. These rules apply repository-wide and override any default assumptions.

I am **no longer using Jupyter Notebooks**. All development is done in **production-grade Python scripts (`.py`)**.

Your objective is to generate **debug-ready, production-quality code** that provides **full execution visibility via logs**, without relying on interactive cells or print statements.

---

## **1. Mandatory Logging Standard — `structlog` (Non-Negotiable)**

### **No `print()` statements**

- Absolutely **no `print()`** usage.
- **All output must be logged** using `structlog`.

### **Configuration Strategy**

You must configure `structlog` based on the context of the request:

**Scenario A: Standalone Script**
If generating a single, self-contained script, **include this exact block at the top**:

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

**Scenario B: Project / Multiple Files**
If generating part of a larger project, make sure configuration is done in a `utils/logger.py` file.

- If the `utils/logger.py` file does not exist, create it and include the configuration block.
- If the `utils/logger.py` file exists, import the logger from it.

### **Event-Based Logging Only**

Logs must be **structured, event-based, and machine-readable**.
❌ **Bad:** `logger.info(f"Processing item {i}")`
✅ **Good:** `logger.info("processing_item", index=i, status="active")`

### **Data Profiling Instead of Printing**

When loading, transforming, or inspecting data (e.g., DataFrames, tensors, arrays, lists, JSON and dictionaries):

- **Do NOT print the data.**
- **Log a profile instead**, including:
- Shape / dimensions
- Column names / Types
- Memory usage
- Distinct / null counts
  _This allows full state verification **via logs only**._

---

## **2. Debugging & Code Structure Requirements**

### **Contextual Logging**

At the start of every major function or workflow, bind context:

```python
log = logger.bind(task="data_cleaning", file_id=file_path)

```

All subsequent logs in that function **must use the bound `log` variable**.

### **Modular, Debug-Friendly Design**

- Break logic into **small, single-purpose functions**.
- **Type Hinting is Mandatory:** All functions must have Python type hints (e.g., `def process_data(df: pd.DataFrame) -> dict:`).
- Code must be friendly to **Step-Through Debugging** (avoid one-liners that do too much).

### **Exception Handling (Never Silent)**

- **Never swallow exceptions.**
- Always log failures using: `log.exception("event_failed")`.
- This automatically captures the stack trace and bound context.

---

## **3. Documentation & Output Standards**

### **Google-Style Docstrings Required**

- Every module, function, and class must have a docstring following the **Google Python Style Guide**.
- **Args:** List parameters and types.
- **Returns:** Describe output.
- **Raises:** List possible exceptions.

### **Output Requirements**

- Always return a **complete, runnable `.py` script**.
- **Dependencies:** At the very top (comment block), list the required `pip install` commands.
- The execution flow **must be traceable entirely through terminal logs**.

---

## **Enforcement Expectation**

If a request violates any rule above:

1. **Refactor the solution to comply** immediately.
2. **Do not ask for permission.**
3. These rules are mandatory by default.
