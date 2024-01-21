FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt
COPY . .
EXPOSE 8502
CMD [ "streamlit", "run" , "cov_app.py"]
