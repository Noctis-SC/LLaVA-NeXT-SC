import gradio as gr
import time
import torch
import numpy as np
from decord import VideoReader, cpu
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.conversation import conv_templates
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
import copy
# Load the model
pretrained = r"/root/autodl-tmp/sanya/LLaVA-NeXT-SC/LLaVA-Video-7B-Qwen2"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"

tokenizer, model, image_processor, max_length = load_pretrained_model(
    pretrained, None, model_name, torch_dtype="bfloat16", device_map=device_map
)
model.eval()

# Function to load video frames
def load_video(video_path, max_frames_num, fps=1, force_sample=False):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps() / fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i / vr.get_avg_fps() for i in frame_idx]

    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i / vr.get_avg_fps() for i in frame_idx]

    frame_time = ", ".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames, frame_time, video_time

# Function to clear outputs and memory
def clear_context():
    torch.cuda.empty_cache()  # Clear unused GPU memory
    return "", "", "Memory cleared!"

def process_video(video_file, max_frames_num=16):
    try:
        # Directly use video_file as path since it's a string
        video, frame_time, video_time = load_video(video_file, max_frames_num, force_sample=True)
        video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().to(torch.bfloat16)
        video = [video]

        conv_template = "qwen_1_5"
        time_instruction = (
            f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. "
            f"These frames are located at {frame_time}. Please answer the following questions related to this video."
        )
        # time_instruction = (f"视频持续{video_time:.2f}秒钟, 从中均匀采样了{len(video[0])}帧."
        #                     f"这些帧位于{frame_time}. 请回答与此视频相关的以下问题."
        #                     )
        print(time_instruction)
        #question = DEFAULT_IMAGE_TOKEN + f"\n{time_instruction}\n请一句话描述视频中角色的特征和活动."
        question = DEFAULT_IMAGE_TOKEN + f"\n{time_instruction}\n""Please describe the characteristics and activities of the characters in the video in one sentence."
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

        start_time = time.time()
        cont = model.generate(
            input_ids,
            images=video,
            modalities=["video"],
            do_sample=False, # Enable sampling
            temperature=0.1, # Control randomness
            top_p=None, # Nuclues sampling
            top_k=None, # Top-k sampling
            max_new_tokens=100,
        )
        text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
        end_time = time.time()
        time_taken = end_time - start_time

        return text_outputs, f"Time taken: {time_taken:.2f} seconds"
    
    except Exception as e:
        return f"Error processing video: {str(e)}", "Processing failed"

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("## Visual Large Language Model Demo")
    
    with gr.Row():
        video_input = gr.Video(label="Upload Video")
        output_text = gr.Textbox(label="Detailed Description", lines=5)
        processing_time = gr.Textbox(label="Processing Time", lines=1)

    submit_button = gr.Button("Analyze Video")
    clear_button = gr.Button("Clear Memory")

    submit_button.click(
        process_video,
        inputs=[video_input],
        outputs=[output_text, processing_time],
    )
    clear_button.click(
        clear_context,
        inputs=[],
        outputs=[output_text, processing_time],
    )

# Launch the Gradio app
if __name__ == "__main__":
    demo.launch(share=False, debug=False)
