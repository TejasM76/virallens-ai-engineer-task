FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app
ENV GEMINI_API_KEY=${GEMINI_API_KEY}

EXPOSE 8000

CMD ["python", "main.py"]