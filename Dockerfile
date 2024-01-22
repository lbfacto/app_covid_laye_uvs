FROM python:3.8
COPY . .
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "cov_app.py"]

