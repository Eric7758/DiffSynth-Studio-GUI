import gradio as gr
import os
import sys
import json
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append('/workspace')

def get_model_paths():
    """获取模型路径列表"""
    model_dirs = [
        '/workspace/models',  # 项目内置模型目录
        './models',          # 当前目录模型
        '/models'            # 通用模型目录
    ]
    
    model_files = []
    for model_dir in model_dirs:
        if os.path.exists(model_dir):
            for root, dirs, files in os.walk(model_dir):
                for file in files:
                    if file.endswith(('.safetensors', '.ckpt', '.bin', '.onnx')):
                        model_files.append(os.path.join(root, file))
    
    return sorted(list(set(model_files)))  # 去重并排序

def get_output_formats():
    """返回支持的输出格式"""
    return ["PNG", "JPEG", "WEBP"]

def get_scheduler_options():
    """返回调度器选项"""
    return ["FlowMatchEulerDiscreteScheduler", "DDPMScheduler", "DPMSolverMultistepScheduler", "EulerDiscreteScheduler", "PNDMScheduler"]

def get_device_options():
    """返回设备选项"""
    return ["cuda", "cpu"]

def train_lora_ui():
    """Lora训练界面"""
    with gr.Tab("LoRA Training"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("## LoRA Training Parameters")
                
                # 数据集参数
                dataset_path = gr.Textbox(label="Dataset Path", placeholder="/path/to/your/dataset")
                resolution = gr.Slider(minimum=256, maximum=2048, value=512, step=64, label="Resolution")
                batch_size = gr.Slider(minimum=1, maximum=32, value=1, step=1, label="Batch Size")
                
                # 模型参数
                model_name = gr.Dropdown(
                    choices=["black-forest-labs/FLUX.1-dev", "stabilityai/stable-diffusion-3-medium", "stabilityai/sdxl-turbo", "runwayml/stable-diffusion-v1-5", "prompthero/openjourney", "stabilityai/stable-diffusion-2-1"],
                    value="black-forest-labs/FLUX.1-dev",
                    label="Model Name"
                )
                
                # 训练参数
                learning_rate = gr.Number(value=1e-4, label="Learning Rate")
                max_train_steps = gr.Slider(minimum=100, maximum=10000, value=1000, step=100, label="Max Train Steps")
                save_steps = gr.Slider(minimum=100, maximum=1000, value=500, step=100, label="Save Steps")
                
                # LoRA特定参数
                lora_rank = gr.Slider(minimum=4, maximum=256, value=16, step=4, label="LoRA Rank")
                lora_alpha = gr.Slider(minimum=1, maximum=256, value=16, step=1, label="LoRA Alpha")
                
                # 输出设置
                output_dir = gr.Textbox(label="Output Directory", placeholder="/path/to/output", value="./output/lora")
                
                # 训练按钮
                train_btn = gr.Button("Start LoRA Training", variant="primary")
            
            with gr.Column():
                gr.Markdown("## Preview & Logs")
                preview_gallery = gr.Gallery(label="Training Previews", show_label=True, elem_id="gallery")
                logs_output = gr.Textbox(label="Training Logs", interactive=False, lines=10)

        def train_lora_fn(dataset_path, resolution, batch_size, model_name, learning_rate, max_train_steps, save_steps, lora_rank, lora_alpha, output_dir):
            # 这里是实际的训练逻辑（简化版）
            logs = f"Starting LoRA training...\nDataset: {dataset_path}\nModel: {model_name}\nResolution: {resolution}\nBatch size: {batch_size}\nLR: {learning_rate}\nSteps: {max_train_steps}\nLoRA rank: {lora_rank}, alpha: {lora_alpha}\nOutput: {output_dir}"
            
            # 模拟一些预览图像
            previews = []  # 实际应用中应包含真实预览图像
            
            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)
            
            return previews, logs
        
        train_btn.click(
            fn=train_lora_fn,
            inputs=[dataset_path, resolution, batch_size, model_name, learning_rate, max_train_steps, save_steps, lora_rank, lora_alpha, output_dir],
            outputs=[preview_gallery, logs_output]
        )

def image_generation_ui():
    """图像生成界面"""
    with gr.Tab("Image Generation"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("## Image Generation Settings")
                
                # 模型选择
                model_path = gr.Dropdown(
                    choices=get_model_paths(),
                    label="Model Path",
                    allow_custom_value=True
                )
                
                # 提示词输入
                prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here...", lines=3)
                negative_prompt = gr.Textbox(label="Negative Prompt", placeholder="Enter negative prompt here...", lines=2)
                
                # 生成参数
                width = gr.Slider(minimum=256, maximum=2048, value=1024, step=64, label="Width")
                height = gr.Slider(minimum=256, maximum=2048, value=1024, step=64, label="Height")
                num_inference_steps = gr.Slider(minimum=1, maximum=100, value=20, step=1, label="Inference Steps")
                guidance_scale = gr.Slider(minimum=1.0, maximum=20.0, value=7.0, step=0.5, label="Guidance Scale")
                
                # 调度器和设备
                scheduler = gr.Dropdown(choices=get_scheduler_options(), value="FlowMatchEulerDiscreteScheduler", label="Scheduler")
                device = gr.Dropdown(choices=get_device_options(), value="cuda", label="Device")
                
                # 随机种子
                seed = gr.Number(value=-1, label="Seed (-1 for random)")
                
                # 生成按钮
                generate_btn = gr.Button("Generate Image", variant="primary")
            
            with gr.Column():
                gr.Markdown("## Generated Images")
                output_image = gr.Image(label="Generated Image", type="filepath")
                download_btn = gr.File(label="Download Generated Image")
        
        def generate_image_fn(model_path, prompt, negative_prompt, width, height, num_inference_steps, guidance_scale, scheduler, device, seed):
            # 这里是实际的生成逻辑（简化版）
            import time
            import numpy as np
            from PIL import Image
            
            # 创建一个模拟图像作为示例
            # 在实际实现中，这里会调用DiffSynth的生成函数
            if seed == -1:
                seed = int(time.time())
                
            # 设置随机种子
            np.random.seed(int(seed))
            
            image_array = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            temp_path = f"/tmp/generated_{int(time.time())}_{seed}.png"
            Image.fromarray(image_array).save(temp_path)
            
            logs = f"Image generated with settings:\nModel: {model_path}\nPrompt: {prompt}\nSize: {width}x{height}\nSteps: {num_inference_steps}\nScale: {guidance_scale}\nScheduler: {scheduler}\nDevice: {device}\nSeed: {seed}"
            
            return temp_path, temp_path, logs
        
        generate_btn.click(
            fn=generate_image_fn,
            inputs=[model_path, prompt, negative_prompt, width, height, num_inference_steps, guidance_scale, scheduler, device, seed],
            outputs=[output_image, download_btn]
        )

def model_management_ui():
    """模型管理界面"""
    with gr.Tab("Model Management"):
        gr.Markdown("## Model Management")
        
        with gr.Row():
            with gr.Column():
                model_type = gr.Radio(
                    choices=["Text Encoder", "UNet", "VAE", "Other"],
                    value="UNet",
                    label="Model Type"
                )
                
                model_upload = gr.File(label="Upload Model File", file_types=[".safetensors", ".ckpt", ".bin"])
                
                model_download_url = gr.Textbox(
                    label="Download Model URL",
                    placeholder="https://huggingface.co/model-name/resolve/main/model.safetensors"
                )
                
                download_btn = gr.Button("Download Model")
            
            with gr.Column():
                available_models = gr.Dropdown(
                    choices=get_model_paths(),
                    label="Available Models",
                    multiselect=True
                )
                
                refresh_btn = gr.Button("Refresh Model List")
                
                delete_btn = gr.Button("Delete Selected Models", variant="stop")
        
        def refresh_models():
            return gr.Dropdown(choices=get_model_paths())
        
        def download_model(url):
            # 简化版本：仅显示下载信息
            return f"Would download model from: {url}"
        
        refresh_btn.click(refresh_models, outputs=available_models)
        download_btn.click(download_model, inputs=model_download_url, outputs=None)

def main():
    """主界面"""
    with gr.Blocks(title="DiffSynth Toolkit GUI") as demo:
        gr.Markdown("# DiffSynth Toolkit GUI")
        gr.Markdown("A user-friendly interface for DiffSynth Toolkit operations")
        
        # 创建标签页
        train_lora_ui()
        image_generation_ui()
        model_management_ui()
        
        gr.Markdown("---")
        gr.Markdown("Powered by DiffSynth Toolkit and Gradio")
    
    # 启动应用
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, debug=True)

if __name__ == "__main__":
    main()