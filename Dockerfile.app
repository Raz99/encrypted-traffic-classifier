# Using a slim version of Python for efficiency
FROM python:3.11-slim

WORKDIR /app

# Install only UI and API requirements
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

# Create non-root user for security
RUN useradd -m appuser && chown -R appuser /app

# Copy application and model files
COPY . .
RUN chown -R appuser /app

# Switch to non-root user
USER appuser

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000
EXPOSE 8501