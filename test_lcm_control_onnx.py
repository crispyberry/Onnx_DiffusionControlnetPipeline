from diffusers.utils import load_image
import cv2
from PIL import Image
import numpy as np
from diffusers import UniPCMultistepScheduler
from diffusers.pipelines.onnx_utils import OnnxRuntimeModel
from pipeline_onnx_stable_diffusion_controlnet import OnnxStableDiffusionControlNetPipeline
import onnxruntime as ort
from diffusers.schedulers import LCMScheduler
from pathlib import Path
from diffusers import UniPCMultistepScheduler
import time



image = load_image("./input_image_vermeer.png")
image = np.array(image)

low_threshold = 100
high_threshold = 200

# TO canny

image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)       

opts = ort.SessionOptions()
opts.enable_cpu_mem_arena = False
opts.enable_mem_pattern = False

# We dont need to load controlnet again.
# controlnet = OnnxRuntimeModel.from_pretrained("./bakmodel/canny_onnx/newer", sess_options=opts, provider="DmlExecutionProvider")

# After convert, put your result as a whole under model. You also need to move scheduler, tonkenizer from original
pipe = OnnxStableDiffusionControlNetPipeline.from_pretrained(
    "../model/lcm-canny", controlnet=None,
    sess_options=opts, 
    provider="DmlExecutionProvider",
)

lcm_config = {
    "beta_start": 0.001,
    "beta_end": 0.0135,
    "beta_schedule": "scaled_linear",
    "timestep_spacing": "linspace",
    "prediction_type": "epsilon",
    "steps_offset": 1
}
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

# # Better choice with UniPCMultistepScheduler
# pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

prompt = "cyberpunk style"
seed = time.time()
generator = np.random.RandomState((int)(seed))

images = pipe(
    prompt,
    canny_image,
    negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
    guidance_scale=2.0, #if you are using UniPCMultistepScheduler, you need to set guidance_scale to 7.5
    num_inference_steps=10,
    controlnet_conditioning_scale=0.5,
    generator=generator,
).images[0]
images.save("test.png")
