FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y unzip && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir gdown

RUN gdown --id 1D36IMkfT23_xFwxeG0wDn9imunBVM19J \
    && unzip model.zip \
    && rm model.zip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8888

CMD ["python", "run_api.py"]