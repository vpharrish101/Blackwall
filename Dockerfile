FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir \
    numpy pandas torch scikit-learn xgboost pyyaml
CMD ["python","run_blackwall.py"]
