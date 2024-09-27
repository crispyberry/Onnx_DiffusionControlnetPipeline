# Onnx_DiffusionControlnetPipeline
To run diffusion-based controlnet with onnxruntime(especially for Windows DirectML Backend). This is inspired by https://github.com/Amblyopius/Stable-Diffusion-ONNX-FP16
<br>The pipeline can only support exported onnx model, which combines diffusion model and controlnet in diffusers before
<br>You can use converted model from https://huggingface.co/ssslvky/lcm-hed-onnx, then try with test_lcm_control_onnx.py(WIP: change with hed by yourself)
<br>Or you can convert with convert_stable_diffusion_controlnet_to_onnx.py.py, which is provided by huggingface diffusers

