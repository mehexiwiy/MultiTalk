from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
import os
from huggingface_hub import snapshot_download
from generate_multitalk import generate_multitalk

app = FastAPI()


@app.on_event("startup")
def startup():
    os.makedirs("checkpoints", exist_ok=True)
    hf_token = os.getenv("HF_TOKEN")  # Safe fallback: None if not set

    # Download models (only once; cached afterward)
    snapshot_download("MeiGen-AI/MeiGen-MultiTalk", local_dir="checkpoints", token=hf_token)
    snapshot_download("MeiGen-AI/Wan2.1", local_dir="checkpoints/wan", token=hf_token)
    snapshot_download("MeiGen-AI/chinese-wav2vec2", local_dir="checkpoints/wav2vec2", token=hf_token)


@app.post("/generate/")
async def generate(
    ref_image: UploadFile = File(...),
    audio: UploadFile = File(...),
    prompt: str = Form(...)
):
    # Prepare temp working directory
    os.makedirs("temp", exist_ok=True)

    img_path = os.path.join("temp", ref_image.filename)
    aud_path = os.path.join("temp", audio.filename)

    # Save uploaded files
    with open(img_path, "wb") as f:
        f.write(await ref_image.read())
    with open(aud_path, "wb") as f:
        f.write(await audio.read())

    # Output path
    output = "output.mp4"

    # Run the model
    generate_multitalk(
        ref_image=img_path,
        audio=aud_path,
        prompt=prompt,
        output=output,
        wav2vec_dir="checkpoints/wav2vec2"
    )

    # Serve back the result
    return FileResponse(output, media_type="video/mp4", filename="generated.mp4")
