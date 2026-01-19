from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from .inference import Stylizer

app = FastAPI(title="Ghibli CycleGAN API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later restrict
    allow_methods=["*"],
    allow_headers=["*"],
)

stylizer = Stylizer(weights_path="weights/G_A2B.pth", num_res_blocks=6)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/stylize")
async def stylize(file: UploadFile = File(...)):
    img_bytes = await file.read()
    out_bytes = stylizer.stylize_bytes(img_bytes)
    return Response(content=out_bytes, media_type="image/png")
