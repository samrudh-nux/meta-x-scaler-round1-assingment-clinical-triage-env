FROM python:3.11-slim

WORKDIR /app


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY . .


RUN mkdir -p server && \
    [ -f server/__init__.py ] || echo "# server package" > server/__init__.py

EXPOSE 7860

# Use root app.py directly — fastest startup, no import chains
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
