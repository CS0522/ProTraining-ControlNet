print("[gradio_canny2image.py] 程序启动")
from share import *
print("[gradio_canny2image.py] share 已经导入")
import config

print("[gradio_canny2image.py] 开始导入依赖库")
import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random

print("[gradio_canny2image.py] 开始导入torch随机种子库")
from pytorch_lightning import seed_everything

print("[gradio_canny2image.py] 导入预处理器工具类 (resize_image, HWC3)")
from annotator.util import resize_image, HWC3

print("[gradio_canny2image.py] 导入Canny预处理方法")
from annotator.canny import CannyDetector

print("[gradio_canny2image.py] 导入模型加载函数 (create_model, load_state_dict)")
from cldm.model import create_model, load_state_dict

print("[gradio_canny2image.py] 导入DDIM采样器")
from cldm.ddim_hacked import DDIMSampler

print("[gradio_canny2image.py] 初始化边缘检测预处理器对象")
apply_canny = CannyDetector()

print("[gradio_canny2image.py] 开始加载ControlNet Canny模型文件")
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict('./models/control_sd15_canny.pth', location='cuda'))
model = model.cuda()

print("[gradio_canny2image.py] ControlNet Canny模型文件加载完成")
ddim_sampler = DDIMSampler(model)
print("[gradio_canny2image.py] DDIM采样方法对象初始化完成")


def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold):
    with torch.no_grad():
        ## 处理图像为设置的分辨率
        print("[gradio_canny2image.py] 开始将输入的图像处理为设置的分辨率")
        img = resize_image(HWC3(input_image), image_resolution)
        H, W, C = img.shape

        ## 检测边缘，将输入图像转为线稿(边缘图)
        print("[gradio_canny2image.py] 开始将输入的图像转化为线稿(边缘图)")
        detected_map = apply_canny(img, low_threshold, high_threshold)
        detected_map = HWC3(detected_map)

        ## TODO: ???
        print("[gradio_canny2image.py] 开始将线稿图转化为控制条件")
        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        ## 设置种子
        print("[gradio_canny2image.py] 开始第一次设置随机种子")
        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        print("[gradio_canny2image.py] 开始第一次读取配置文件是否为低显存模型运行")
        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        ## TODO 加载参数 ???
        ## 无提示词模式开启时ControlNet不处理反向提示词，此时反向提示词只经过sd
        ## --> [File cldm/cldm.py Line 334]: if cond['c_concat'] is None:
        print("[gradio_canny2image.py] 开始根据是否启用无提示词模式，确定提示词和反向提示词由谁处理")
        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        print("[gradio_canny2image.py] 开始第二次读取配置文件是否为低显存模型运行")
        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        ## 设置控制权重(包括无提示词模式) 使用DDIM采样器采样生成
        ## 无提示词模式未开启时，控制权重为固定值，否则ControlNet部分13层每层的结构权重递增，范围0到1。
        print("[gradio_canny2image.py] 开始设置ControlNet的控制权重")
        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        print("[gradio_canny2image.py] 开始运行DDIM采样器")
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        print("[gradio_canny2image.py] 开始第三次读取配置文件是否为低显存模型运行")
        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        ## VAE 解码 ???
        print("[gradio_canny2image.py] 开始解码生成的图像")
        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        print("[gradio_canny2image.py] 开始处理生成结果")
        results = [x_samples[i] for i in range(num_samples)]
        print("[gradio_canny2image.py] 返回生成结果")
    return [255 - detected_map] + results

print("[gradio_canny2image.py] 加载WebUI网页界面")
block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## Control Stable Diffusion with Canny Edge Maps")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source='upload', type="numpy")
            prompt = gr.Textbox(label="Prompt")
            run_button = gr.Button(label="Run")
            with gr.Accordion("Advanced options", open=False):
                num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
                image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
                strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                guess_mode = gr.Checkbox(label='Guess Mode', value=False)
                low_threshold = gr.Slider(label="Canny low threshold", minimum=1, maximum=255, value=100, step=1)
                high_threshold = gr.Slider(label="Canny high threshold", minimum=1, maximum=255, value=200, step=1)
                ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
                scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
                seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
                eta = gr.Number(label="eta (DDIM)", value=0.0)
                a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed')
                n_prompt = gr.Textbox(label="Negative Prompt",
                                      value='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality')
        with gr.Column():
            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')
    ips = [input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold]
    run_button.click(fn=process, inputs=ips, outputs=[result_gallery])

print("[gradio_canny2image.py] 启动WebUI")
block.launch(server_name='0.0.0.0')
