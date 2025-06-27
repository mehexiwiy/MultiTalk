FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git ffmpeg build-essential python3 python3-pip python3-dev cmake libgl1 libglib2.0-0 \
    && apt-get clean

RUN python3 -m pip install --upgrade pip

# ✅ Install only necessary packages
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip install fastapi uvicorn huggingface_hub

# If you’re using a requirements.txt:
# COPY requirements.txt .
# RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
