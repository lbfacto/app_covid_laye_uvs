FROM python:3.8
COPY . .
WORKDIR /app
RUN python -m venv envcov
RUN pip install r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "cov_app.py"]

