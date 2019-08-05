FROM aaftio/face_recognition

COPY ./ ./app
WORKDIR ./app

RUN pip3 install -r requirements.txt

ENTRYPOINT ["python3"]
CMD ["face_detection.py"]