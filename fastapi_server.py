import io
import torch
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import ChatTTS
import logging

# 配置日志，方便排查问题
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ChatTTS API Server")

# --- 关键修复 1: 严格的 CORS 配置 ---
# 允许所有来源，确保前端无论用什么端口都能访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局初始化模型
print("start ChatTTS...")
chat = ChatTTS.Chat()
chat.load(compile=False)

class TTSRequest(BaseModel):
    text: str

@app.post("/api/tts")
async def generate_tts(request: TTSRequest):
    logger.info(f"收到合成请求，文本长度: {len(request.text)}")

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="文本不能为空")

    try:
        # 生成音频 (wav_infer 返回的是 numpy array 列表)
        wavs = chat.infer(request.text)

        if wavs is None or len(wavs) == 0:
            raise HTTPException(status_code=500, detail="模型生成失败")

        # 获取第一段音频
        wav_data = wavs[0]

        # --- 关键修复 2: 在内存中处理 WAV 文件 ---
        buffer = io.BytesIO()
        # sample_rate 通常是 24000 (ChatTTS 默认)
        sf.write(buffer, wav_data, samplerate=24000, format='WAV')
        buffer.seek(0)

        logger.info(f"音频生成成功，大小: {len(buffer.getvalue())} bytes")

        # 使用 StreamingResponse 返回，设置 media_type 为 audio/wav
        return StreamingResponse(
            buffer,
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=speech.wav"
            }
        )

    except Exception as e:
        logger.error(f"合成出错: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # 明确指定 host 和 port
    uvicorn.run(app, host="0.0.0.0", port=8000)