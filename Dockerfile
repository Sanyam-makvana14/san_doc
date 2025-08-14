FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip &&     pip install -r requirements.txt

ENTRYPOINT ["python", "entrypoint.py"]
