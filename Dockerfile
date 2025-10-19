FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
COPY data/ /app/data/
COPY models/ /app/models/
COPY Frontend/ /app/Frontend/

EXPOSE 5000

CMD ["python", "app.py"]