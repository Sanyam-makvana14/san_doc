FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

COPY . /app
RUN apt-get update && apt-get install -y wget && \
    mkdir -p results && \
    wget -O results/model_best.pth "https://drive.google.com/uc?export=download&id=1QLf7DI4sRlebzj-zY40K16lz3LcWayL6"

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

ENTRYPOINT ["python", "entrypoint.py"]
