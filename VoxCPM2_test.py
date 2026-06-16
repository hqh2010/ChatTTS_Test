import os
# 强制离线模式（避免任何联网尝试）
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# 使用官方推荐的导入方式
from voxcpm import VoxCPM
import soundfile as sf

# VoxCPM2_local下载地址 https://huggingface.co/openbmb/VoxCPM2/tree/main
# VoxCPM2 占用内存很大，在虚拟机上运行时被系统杀了
# 本地模型路径
local_model_path = "/home/uos/Desktop/ChatTTS-0.2.5/VoxCPM2_local"

# 加载模型
model = VoxCPM.from_pretrained(
    local_model_path,
    load_denoiser=False,      # 禁用降噪器，减少内存占用
    local_files_only=True,    # 强制仅使用本地文件
)

# 生成语音
print("正在生成语音...")
wav = model.generate(
    text="VoxCPM2 是目前推荐使用的多语言语音合成版本。",
    cfg_value=2.0,            # 控制生成质量，越高越稳定
    inference_timesteps=10,   # 推理步数，越低越快
)

# 保存音频
sf.write("demo.wav", wav, model.tts_model.sample_rate)
print("save done：demo.wav")
