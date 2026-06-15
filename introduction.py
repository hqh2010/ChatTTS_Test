import ChatTTS
import torch
import torchaudio

chat = ChatTTS.Chat()
chat.load(compile=False) # Set to True for better performance

texts = ["PUT YOUR 1st TEXT HERE", "PUT YOUR 2nd TEXT HERE"]

wavs = chat.infer(texts)

for i in range(len(wavs)):
    """
    In some versions of torchaudio, the first line works but in other versions, so does the second line.
    """
    try:
        torchaudio.save(f"basic_output{i}.wav", torch.from_numpy(wavs[i]).unsqueeze(0), 24000)
    except:
        torchaudio.save(f"basic_output{i}.wav", torch.from_numpy(wavs[i]), 24000)

###################################
# Sample a speaker from Gaussian.

rand_spk = chat.sample_random_speaker()
#print(rand_spk) # save it for later timbre recovery

params_infer_code = ChatTTS.Chat.InferCodeParams(
    spk_emb = rand_spk, # add sampled speaker 
    temperature = .3,   # using custom temperature
    top_P = 0.7,        # top P decode
    top_K = 20,         # top K decode
)

###################################
# For sentence level manual control.

# use oral_(0-9), laugh_(0-2), break_(0-7) 
# to generate special token in text to synthesize.
params_refine_text = ChatTTS.Chat.RefineTextParams(
    prompt='[oral_2][laugh_0][break_6]',
)

wavs = chat.infer(
    texts,
    params_refine_text=params_refine_text,
    params_infer_code=params_infer_code,
)

###################################
# For word level manual control.

text = 'What is [uv_break]your favorite english food?[laugh][lbreak]'
wavs = chat.infer(text, skip_refine_text=True, params_refine_text=params_refine_text,  params_infer_code=params_infer_code)
"""
In some versions of torchaudio, the first line works but in other versions, so does the second line.
"""
try:
    torchaudio.save("word_level_output.wav", torch.from_numpy(wavs[0]).unsqueeze(0), 24000)
except:
    torchaudio.save("word_level_output.wav", torch.from_numpy(wavs[0]), 24000)


inputs_en1 = """
chat T T S is a text to speech model designed for dialogue applications. 
[uv_break]it supports mixed language input [uv_break]and offers multi speaker 
capabilities with precise control over prosodic elements like 
[uv_break]laughter[uv_break][laugh], [uv_break]pauses, [uv_break]and intonation. 
[uv_break]it delivers natural and expressive speech,[uv_break]so please
[uv_break] use the project responsibly at your own risk.[uv_break]
""".replace('\n', '') # English is still experimental.

inputs_en = """
四川美食以辣闻名，但也有不辣的选择。
我的家乡湖北江汉平源，有鱼米之乡之称的云梦。
大家好，我来自开发二组，爱好乒乓球，游泳。
""".replace('\n', '') # English is still experimental.

params_refine_text = ChatTTS.Chat.RefineTextParams(
    prompt='[oral_2][laugh_0][break_4]',
)

audio_array_en = chat.infer(inputs_en, params_refine_text=params_refine_text)
# torchaudio.save("self_introduction_output.wav", torch.from_numpy(audio_array_en[0]), 24000)
try:
    torchaudio.save("self_introduction_output.wav", torch.from_numpy(audio_array_en[0]).unsqueeze(0), 24000)
except:
    torchaudio.save("self_introduction_output.wav", torch.from_numpy(audio_array_en[0]), 24000)