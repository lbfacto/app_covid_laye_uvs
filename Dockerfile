FROM python:3.8

WORKDIR /app

COPY . .

RUN ls

RUN python -m venv envcov
RUN pip install -r requirements.txt

EXPOSE 3535

CMD ["streamlit", "run", "cov_app.py"]
