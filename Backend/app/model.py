import torch
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, ch: int, use_dropout: bool = False):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch, ch, 3, 1, 0, bias=True),
            nn.InstanceNorm2d(ch, affine=False),
            nn.ReLU(inplace=True),
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        layers += [
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch, ch, 3, 1, 0, bias=True),
            nn.InstanceNorm2d(ch, affine=False),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 3, num_res_blocks: int = 6, base_ch: int = 64):
        super().__init__()

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, base_ch, 7, 1, 0, bias=True),
            nn.InstanceNorm2d(base_ch, affine=False),
            nn.ReLU(inplace=True),
        ]

        model += [
            nn.Conv2d(base_ch, base_ch * 2, 3, 2, 1, bias=True),
            nn.InstanceNorm2d(base_ch * 2, affine=False),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_ch * 2, base_ch * 4, 3, 2, 1, bias=True),
            nn.InstanceNorm2d(base_ch * 4, affine=False),
            nn.ReLU(inplace=True),
        ]

        for _ in range(num_res_blocks):
            model.append(ResBlock(base_ch * 4, use_dropout=False))

        model += [
            nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 3, 2, 1, output_padding=1, bias=True),
            nn.InstanceNorm2d(base_ch * 2, affine=False),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(base_ch * 2, base_ch, 3, 2, 1, output_padding=1, bias=True),
            nn.InstanceNorm2d(base_ch, affine=False),
            nn.ReLU(inplace=True),
        ]

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(base_ch, out_ch, 7, 1, 0, bias=True),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


def load_generator(weights_path: str, device: str = "cpu", num_res_blocks: int = 6, strict: bool = False):
    model = Generator(3, 3, num_res_blocks=num_res_blocks).to(device)
    state = torch.load(weights_path, map_location=device)

    incompat = model.load_state_dict(state, strict=strict)
    missing = incompat.missing_keys
    unexpected = incompat.unexpected_keys

    if missing or unexpected:
        print(" Weight load warnings:")
        if missing:
            print("Missing keys:", missing[:10], "..." if len(missing) > 10 else "")
        if unexpected:
            print("Unexpected keys:", unexpected[:10], "..." if len(unexpected) > 10 else "")
        print("This usually means your architecture doesn't match training.")
    else:
        print("Weights loaded perfectly")

    model.eval()
    return model
