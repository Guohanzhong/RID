import torch
from libs.DiT import DiT
from thop import profile, clever_format

# Initialize the model
noise_models = DiT(scale=12/255,patch_size=8,depth=26)

# Example input
input = torch.randn(1, 3, 512, 512)

# Profile the original model
macs, params = profile(noise_models, inputs=(input,))
macs, params = clever_format([macs, params], "%.3f")
print(f"Original Model: params(M) is {params}, MACs(G) is {macs}")

# Quantize the model
model_int8 = torch.quantization.convert(noise_models)

# Profile the quantized model
# Ensure the input tensor is float for the quantized model
quantized_input = input.to(torch.float32)
macs, params = profile(model_int8, inputs=(quantized_input,))

macs, params = clever_format([macs, params], "%.3f")
print(f"Quantized Model: params(M) is {params}, MACs(G) is {macs}")