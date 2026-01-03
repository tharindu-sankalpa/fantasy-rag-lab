# **Global Code Generation Rules (Mandatory)**

From this point onward, **all generated code, scripts, and Python programs must strictly follow the rules below**. These rules apply repository-wide and override any default assumptions.

I am **no longer using Jupyter Notebooks**. All development is done in **production-grade Python scripts (`.py`)**.

Your objective is to generate **debug-ready, production-quality code** that is **highly readable and transparent**. The code must provide full execution visibility via logs and be immediately understandable to a human reader through simplicity and extensive commentary.

---

## **1. Mandatory Logging Standard — `structlog` (Non-Negotiable)**

### **No `print()` statements**

- Absolutely **no `print()**` usage.
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

When loading, transforming, or inspecting data (e.g., DataFrames, tensors, arrays, lists, JSON, and dictionaries):

- **Do NOT print the data.**
- **Log a profile instead**, including:
- Shape / dimensions
- Column names / Types
- Memory usage
- Distinct / null counts
  _This allows full state verification **via logs only**._

---

## **2. Architecture, Simplicity & Anti-Over-Engineering**

### **Radical Readability & Simplicity**

- **Avoid Over-Engineering:** Do not use complex design patterns (e.g., Decorators, Factories, Metaclasses) unless strictly necessary for functionality.
- **Explicit > Implicit:** Logic must be immediately visible. Do not hide behavior behind obscure abstractions.
- **Linear Flow:** Prefer linear, procedural logic within functions over deeply nested structures or recursive complexity.
- **Goal:** A developer should be able to read the code once and understand exactly what it does without jumping between multiple files or classes unnecessarily.

### **Modular, Debug-Friendly Design**

- Break logic into **small, single-purpose functions**.
- **Type Hinting is Mandatory:** All functions must have Python type hints (e.g., `def process_data(df: pd.DataFrame) -> dict:`).
- Code must be friendly to **Step-Through Debugging** (avoid one-liners that do too much).

### **Contextual Logging**

At the start of every major function or workflow, bind context:

```python
log = logger.bind(task="data_cleaning", file_id=file_path)

```

All subsequent logs in that function **must use the bound `log` variable**.

### **Exception Handling (Never Silent)**

- **Never swallow exceptions.**
- Always log failures using: `log.exception("event_failed")`.
- This automatically captures the stack trace and bound context.

---

## **3. Documentation, Comments & Output Standards**

### **High-Resolution Docstrings**

- Every module, function, and class must have a **Google-Style Docstring**.
- **Tone:** Professional, yet highly explanatory.
- **Content:** Do not just describe _what_ the function does, but _how_ it fits into the broader workflow.
- **Args/Returns/Raises:** Must be exhaustively detailed.

### **Dense, Line-by-Line Commentary**

- **Code must be self-narrating.**
- **Algorithmic/Complex Logic:** You must provide **line-by-line comments** explaining the functionality.
- _Example:_ If performing a tensor operation or complex DataFrame filter, explain why this specific operation is happening right above the line.

- **Business Logic:** Comments should explain the "Why" behind the code, ensuring the intent is clear to future maintainers.
- **Visual Scannability:** Use whitespace and comment blocks to visually separate logical steps within a function.

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
