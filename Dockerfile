FROM python:3.11.3-slim-buster

WORKDIR /app

COPY . /app

RUN pip install -r requirements.lock

ENV GOOGLE_APPLICATION_CREDENTIALS="/app/.secrets/key.json"

EXPOSE 8501

ENTRYPOINT ["streamlit", "run"]

CMD ["main.py"]
