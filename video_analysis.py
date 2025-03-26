from openai import OpenAI
import base64
from decord import VideoReader, cpu
import numpy as np
from PIL import Image
from io import BytesIO
import logging
import os
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
import copy
import torch
import warnings
import torch
from torchvision import transforms

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_video(video_path, max_frames_num=16, fps=1, force_sample=True):
    if not isinstance(video_path, str):
        raise TypeError(f"Expected video_path to be a string, but got {type(video_path)}")
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
        
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3)), "0s", 0
        
    fps = round(vr.get_avg_fps()/fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i/fps for i in frame_idx]
    
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i/vr.get_avg_fps() for i in frame_idx]
        
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames, frame_time, video_time


def process_video(video_path, prompt):
    try:
        # Load and process video frames
        video, frame_time, video_time = load_video(video_path, max_frames_num=16)
        
        # Setup transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            lambda x: x.to(torch.bfloat16)
        ])
        
        # Process frames to base64 URLs
        frame_urls = []
        for frame in video:
            # Convert to tensor and ensure bfloat16
            img = Image.fromarray(frame)
            img_tensor = transform(img)
            
            # Convert back to numpy/PIL for base64 encoding
            img_numpy = img_tensor.float().numpy().transpose(1, 2, 0)
            img_pil = Image.fromarray((img_numpy * 255).astype(np.uint8))
            
            buffered = BytesIO()
            img_pil.save(buffered, format="JPEG", quality=50)
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            frame_urls.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_str}"
                }
            })

        # Rest of your existing code...
        client = OpenAI(base_url="http://localhost:9000/v1", api_key="None")
        
        
        time_instruction = f"The video lasts for {video_time:.2f} seconds, and {len(frame_urls)} frames are uniformly sampled from it. These frames are located at {frame_time}."
        print(time_instruction)
        content = frame_urls + [{
            "type": "text",
            "text": f"{DEFAULT_IMAGE_TOKEN}{time_instruction}\n{prompt}"
        }]
        
        response = client.chat.completions.create(
            model="/root/autodl-tmp/sanya/LLaVA-NeXT-SC/SanyaChoi/LLaVA-Video-7B-Qwen2-hf",
            messages=[{
                "role": "user",
                "content": content
            }],
            temperature=0.2,
            max_tokens=200
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        return f"Error: {str(e)}"
    
if __name__ == "__main__":
    video_path = "/root/autodl-tmp/sanya/LLaVA-NeXT-SC/package_videos/istockphoto-1084227724-640_adpp_is.mp4"
    # prompt = "Please briefly describe this video characters involved and their actions."
    prompt = "Please briefly describe the video"
    result = process_video(video_path, prompt)
    print(result)