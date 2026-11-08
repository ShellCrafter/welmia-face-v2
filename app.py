
import io
import os
import sys
import json
import torch

from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

app = FastAPI(title="Welmia Face 2.0 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_ID = "Welmia/welmia-face-2.0-7.6m-base"

generator = None
config = None

torch.set_num_threads(1)


@app.on_event("startup")
def load_welmia():
    global generator, config

    print("Downloading Welmia Face 2.0...")

    path = snapshot_download(
        repo_id=MODEL_ID,
        allow_patterns=[
            "model.safetensors",
            "welmia_gan.py",
            "welmia_cfg.json",
        ],
    )

    sys.path.insert(0, path)

    import welmia_gan as W

    with open(os.path.join(path, "welmia_cfg.json")) as f:
        config = json.load(f)

    generator = W.make_G(config)

    weights = load_file(
        os.path.join(path, "model.safetensors"),
        device="cpu",
    )

    generator.load_state_dict(weights)
    generator.eval()

    print("Welmia Face 2.0 loaded successfully!")


@app.get("/")
def home():
    return {
        "name": "Welmia Face 2.0",
        "status": "online",
        "model": MODEL_ID,
        "resolution": "128x128"
    }


@app.get("/health")
def health():
    return {
        "status": "ready" if generator is not None else "loading"
    }


@app.get("/generate")
def generate(
    psi: float = 0.7,
    seed: int = None
):
    if generator is None:
        raise HTTPException(503, "Model is loading")

    if not 0.1 <= psi <= 1.0:
        raise HTTPException(400, "psi must be between 0.1 and 1.0")

    try:
        if seed is not None:
            torch.manual_seed(seed)

        z = torch.randn(1, config["z_dim"])

        with torch.inference_mode():
            output = generator(z, psi=psi)
            output = output.clamp(-1, 1) * 0.5 + 0.5

        array = (
            output[0]
            .mul(255)
            .byte()
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )

        image = Image.fromarray(array, mode="RGB")

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)

        return StreamingResponse(
            buffer,
            media_type="image/png",
            headers={
                "Cache-Control": "no-store"
            }
        )

    except Exception as e:
        raise HTTPException(500, str(e))
