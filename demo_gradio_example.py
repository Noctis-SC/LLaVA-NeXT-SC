# import gradio as gr
# import torch
# import numpy as np
# from decord import VideoReader, cpu
# from llava.model.builder import load_pretrained_model
# from llava.mm_utils import tokenizer_image_token
# from llava.conversation import conv_templates
# from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
# import copy
# import warnings
# warnings.filterwarnings("ignore")

# def load_video(video_path, max_frames_num=16, fps=1, force_sample=False):
#     """Load and preprocess video frames"""
#     vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
#     total_frame_num = len(vr)
#     video_time = total_frame_num / vr.get_avg_fps()
#     fps = round(vr.get_avg_fps() / fps)
#     frame_idx = [i for i in range(0, len(vr), fps)]
#     frame_time = [i / vr.get_avg_fps() for i in frame_idx]

#     if len(frame_idx) > max_frames_num or force_sample:
#         sample_fps = max_frames_num
#         uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
#         frame_idx = uniform_sampled_frames.tolist()
#         frame_time = [i / vr.get_avg_fps() for i in frame_idx]

#     frame_time = ", ".join([f"{i:.2f}s" for i in frame_time])
#     spare_frames = vr.get_batch(frame_idx).asnumpy()
#     return spare_frames, frame_time, video_time

# class VideoAnalyzer:
#     def __init__(self):
#         self.pretrained = r"/root/autodl-tmp/sanya/LLaVA-NeXT-SC/LLaVA-Video-7B-Qwen2"
#         self.model_name = "llava_qwen"
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         print(f"Using device: {self.device}")
#         self.load_model()

#     def load_model(self):
#         """Initialize the model"""
#         self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
#             self.pretrained, 
#             None, 
#             self.model_name, 
#             torch_dtype="bfloat16", 
#             device_map="auto"
#         )
#         self.model.eval()
#         print("Model loaded successfully")

#     def analyze_video(self, video_path):
#         """Process and analyze the video"""
#         try:
#             # Load and preprocess video
#             video, frame_time, video_time = load_video(video_path, max_frames_num=64)
#             video = self.image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().to(torch.bfloat16)
#             video = [video]

#             # Prepare prompt
#             time_instruction = f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}. Please answer the following questions related to this video."
#             question = DEFAULT_IMAGE_TOKEN + f"\n{time_instruction}\nPlease describe the characteristics and activities of the characters in the video in one sentence."

#             # Set up conversation
#             conv = copy.deepcopy(conv_templates["qwen_1_5"])
#             conv.append_message(conv.roles[0], question)
#             conv.append_message(conv.roles[1], None)
#             prompt = conv.get_prompt()

#             # Generate response
#             input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)

#             output = self.model.generate(
#                 input_ids,
#                 images=video,
#                 modalities=["video"],
#                 do_sample=False,
#                 temperature=0,
#                 max_new_tokens=200,
#             )

#             response = self.tokenizer.batch_decode(output, skip_special_tokens=True)[0].strip()
#             return response

#         except Exception as e:
#             return f"Error processing video: {str(e)}"

# def create_interface():
#     analyzer = VideoAnalyzer()
    
#     def process_video(video_path):
#         if not video_path:
#             return "Please upload a video first."
#         return analyzer.analyze_video(video_path)

#     # Create Gradio interface with examples
#     demo = gr.Interface(
#         fn=process_video,
#         inputs=gr.Video(label="Upload Video"),
#         outputs=gr.Textbox(label="Video Description", lines=3),
#         title="Video Analysis",
#         description="Upload a video to get an AI-powered description of its content.",
#         # Point these to actual video files inside the `examples` folder
#         examples=[
#             "/root/autodl-tmp/sanya/LLaVA-NeXT-SC/examples/5631-183849543_large.mp4",
#             "/root/autodl-tmp/sanya/LLaVA-NeXT-SC/examples/22835-330970621_tiny.mp4",
#             "/root/autodl-tmp/sanya/LLaVA-NeXT-SC/examples/156318-812205657_large.mp4",
#             "/root/autodl-tmp/sanya/LLaVA-NeXT-SC/examples/istockphoto-1281106686-640_adpp_is.mp4",
#             "/root/autodl-tmp/sanya/LLaVA-NeXT-SC/examples/istockphoto-1318447934-640_adpp_is.mp4"

#         ],
#         cache_examples=False,
#     )
#     return demo

# if __name__ == "__main__":
#     demo = create_interface()
#     demo.launch(
#         server_name="0.0.0.0", 
#         server_port=7860,
#         share=True,
#         debug=True
#     )


import gradio as gr
import torch
import numpy as np
from decord import VideoReader, cpu
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.conversation import conv_templates
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
import copy
import warnings
from pathlib import Path
import time
warnings.filterwarnings("ignore")

def load_video(video_path, max_frames_num=16, fps=1, force_sample=False):
    """Load and preprocess video frames"""
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

class VideoAnalyzer:
    def __init__(self):
        self.pretrained = "/root/autodl-tmp/sanya/LLaVA-NeXT-SC/LLaVA-Video-7B-Qwen2"
        self.model_name = "llava_qwen"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.load_model()

    def load_model(self):
        """Initialize the model"""
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            self.pretrained, 
            None, 
            self.model_name, 
            torch_dtype="bfloat16", 
            device_map="auto"
        )
        self.model.eval()
        print("Model loaded successfully")

    def analyze_video(self, video_path):
        """Process and analyze the video"""
        if not video_path:
            return None, "Please upload a video first.", "Waiting for input..."
            
        try:
            print(f"Processing video: {video_path}")  # Debug print
            processing_start = time.time()
            
            # Load and preprocess video
            video, frame_time, video_time = load_video(video_path, max_frames_num=16)
            video = self.image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().to(torch.bfloat16)
            video = [video]

            # Prepare prompt
            time_instruction = f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}. Please answer the following questions related to this video."
            question = DEFAULT_IMAGE_TOKEN + f"\n{time_instruction}\nPlease briefly describe what is happening in this video. Please describe important information in one sentece."

            # Set up conversation
            conv = copy.deepcopy(conv_templates["qwen_1_5"])
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            # Generate response
            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
            
            output = self.model.generate(
                input_ids,
                images=video,
                modalities=["video"],
                do_sample=False,
                temperature=0.1,
                max_new_tokens=200,
            )

            response = self.tokenizer.batch_decode(output, skip_special_tokens=True)[0].strip()
            processing_time = time.time() - processing_start
            
            print(f"Generated response: {response}")  # Debug print
            return response, "Analysis complete!", f"Processing time: {processing_time:.2f} seconds"

        except Exception as e:
            print(f"Error processing video: {str(e)}")  # Debug print
            return None, f"Error processing video: {str(e)}", "Error occurred!"

def create_interface():
    analyzer = VideoAnalyzer()
    
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo", neutral_hue="slate")) as demo:
        gr.Markdown("""
        # 🎥 Video Analysis AI
        Upload a video to get an AI-powered description of its content.
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                video_input = gr.Video(
                    label="Upload Video",
                )
                
            with gr.Column(scale=3):
                output_text = gr.Textbox(
                    label="Analysis Results",
                    placeholder="Video description will appear here...",
                    lines=4
                )
                status = gr.Textbox(
                    label="Status",
                    value="Ready to analyze video...",
                    lines=1
                )
                time_taken = gr.Textbox(
                    label="Processing Time",
                    lines=1
                )

        with gr.Row():
            analyze_btn = gr.Button("🔍 Analyze Video", variant="primary")
            clear_btn = gr.Button("🗑️ Clear", variant="secondary")

        def clear():
            return None, "", "Ready to analyze video...", ""

        # Set up event handlers
        analyze_btn.click(
            fn=analyzer.analyze_video,
            inputs=video_input,
            outputs=[output_text, status, time_taken],
        )
        
        clear_btn.click(
            fn=clear,
            outputs=[video_input, output_text, status, time_taken],
        )

        # Add examples if available
        if Path("examples").exists():
            example_videos = list(Path("examples").glob("*.mp4"))
            if example_videos:
                gr.Examples(
                    examples=[[str(video)] for video in example_videos],
                    inputs=video_input,
                    outputs=[output_text, status, time_taken],
                    fn=analyzer.analyze_video,
                    cache_examples=False,
                )

    return demo

def setup_examples():
    """Create examples directory"""
    examples_dir = Path("examples")
    examples_dir.mkdir(exist_ok=True)
    print(f"Examples directory created at: {examples_dir.absolute()}")

if __name__ == "__main__":
    # Setup examples directory
    setup_examples()
    
    # Create and launch the interface
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )