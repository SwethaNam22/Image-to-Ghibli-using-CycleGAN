import io
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from .model import Generator

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
])

def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    t = t.squeeze(0).detach().cpu()
    t = (t * 0.5 + 0.5).clamp(0,1)
    arr = (t.permute(1,2,0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)

class Stylizer:
    def __init__(self, weights_path: str, num_res_blocks: int = 6):
        self.model = Generator(3, 3, num_res_blocks=num_res_blocks).to(DEVICE)
        state = torch.load(weights_path, map_location=DEVICE)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

    @torch.no_grad()
    def stylize_bytes(self, image_bytes: bytes) -> bytes:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        x = preprocess(img).unsqueeze(0).to(DEVICE)
        y = self.model(x)
        out = tensor_to_pil(y)

        buf = io.BytesIO()
        out.save(buf, format="PNG")
        return buf.getvalue()
