FROM rasa/rasa_nlu:latest-spacy

RUN pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-1.2.0/en_core_web_sm-1.2.0.tar.gz --no-cache-dir > /dev/null \
    && python -m spacy link en_core_web_sm en 

RUN pip install rasa_core

RUN pip install spacy==1.8.2

RUN pip install flask

VOLUME ["/app/projects", "/app/logs", "/app/data" , "/app/configs"]

EXPOSE 5000 5005

CMD ["python", "./bot.py", "--port=$PORT"]
