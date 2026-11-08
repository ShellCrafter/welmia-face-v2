import base64
from typing import List, Optional
from pydantic import BaseModel

class BatchRequest(BaseModel):
    count: int = 8          # 1..20
    psi: float = 0.7        # 0.1..1.0
    seed: Optional[int] = None   # None = random per image

@app.post("/generate_batch")
def generate_batch(req: BatchRequest):
    if generator is None:
        raise HTTPException(503, "Model is loading")

    if not (1 <= req.count <= 20):
        raise HTTPException(400, "count must be between 1 and 20")
    if not (0.1 <= req.psi <= 1.0):
        raise HTTPException(400, "psi must be between 0.1 and 1.0")

    images = []
    used_seeds = []

    try:
        with torch.inference_mode():
            for i in range(req.count):
                if req.seed is not None:
                    s = req.seed + i          # deterministic sequence
                else:
                    s = int(torch.randint(0, 2**31 - 1, (1,)).item())

                torch.manual_seed(s)

                z = torch.randn(1, config["z_dim"])
                out = generator(z, psi=req.psi)
                out = out.clamp(-1, 1) * 0.5 + 0.5

                arr = (
                    out[0].mul(255).byte()
                    .permute(1, 2, 0).cpu().numpy()
                )
                img = Image.fromarray(arr, mode="RGB")

                buf = io.BytesIO()
                img.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode("ascii")

                images.append(f"data:image/png;base64,{b64}")
                used_seeds.append(s)

        return {"images": images, "seeds": used_seeds, "psi": req.psi}

    except Exception as e:
        raise HTTPException(500, str(e))
