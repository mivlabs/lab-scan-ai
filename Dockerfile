FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-rus fonts-dejavu-core fonts-liberation \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml ./
COPY labscan ./labscan
COPY synth ./synth
COPY evals ./evals
RUN pip install --no-cache-dir .

ENV LABSCAN_DATA=/data LABSCAN_DB=sqlite:////data/labscan.db
VOLUME /data
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')"
CMD ["uvicorn", "labscan.web.app:app", "--host", "0.0.0.0", "--port", "8000"]
