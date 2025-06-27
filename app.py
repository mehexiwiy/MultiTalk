from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
import os
from huggingface_hub import snapshot_download
from generate_multitalk import generate_multitalk

app = FastAPI()

@app.on_event("startup")
def startup():
    os.makedirs("checkpoints", exist_ok=True)
    snapshot_download("MeiGen-AI/MeiGen-MultiTalk", local_dir="checkpoints")
    snapshot_download("MeiGen-AI/Wan2.1", local_dir="checkpoints/wan")
    snapshot_download("MeiGen-AI/chinese-wav2vec2", local_dir="checkpoints/wav2vec2")

@app.post("/generate/")
async def generate(
    ref_image: UploadFile = File(...),
    audio: UploadFile = File(...),
    prompt: str = Form(...)
):
    os.makedirs("temp", exist_ok=True)
    img_path = os.path.join("temp", ref_image.filename)
    aud_path = os.path.join("temp", audio.filename)
    with open(img_path, "wb") as f: f.write(await ref_image.read())
    with open(aud_path, "wb") as f: f.write(await audio.read())

    output = "output.mp4"
    generate_multitalk(
        ref_image=img_path,
        audio=aud_path,
        prompt=prompt,
        output=output,
        wav2vec_dir="checkpoints/wav2vec2"
    )

    return FileResponse(output, media_type="video/mp4", filename="result.mp4")
