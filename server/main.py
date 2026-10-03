from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel
import numpy as np
from groq import Groq
import wave
from dotenv import load_dotenv
from silero_vad import load_silero_vad
from piper import PiperVoice
import torch
app = FastAPI()
vad_model = load_silero_vad()
print("VAD model loaded!")

model = WhisperModel("base", device="cpu", compute_type="int8")
print("Whisper model loaded!")

load_dotenv()
print("Environment Variables loaded")
client = Groq()

piper_voice = PiperVoice.load("F:/Projects/voice-agent/voices/en_US-lessac-medium.onnx")
print("Piper loaded!")


@app.websocket("/ws")
async def audio_echo(websocket: WebSocket):
    await websocket.accept()
    buffer = b""
    vad_buffer = np.array([], dtype=np.float32)
    is_speaking = False
    silence_count= 0  

    transcribed_text = ""
    
    try:
        while True:
            data = await websocket.receive_bytes()
            buffer += data

            batch_samples = np.frombuffer(data, dtype=np.float32) 
            batch_down = batch_samples[::3] 
            vad_buffer = np.concatenate([vad_buffer, batch_down])

            
            
            while len(vad_buffer) >= 512:
                window = vad_buffer[:512]              
                speech_prob = vad_model(torch.from_numpy(window), 16000).item()
                vad_buffer = vad_buffer[512:]

                if speech_prob > 0.5:
                    is_speaking = True
                    silence_count = 0
                else:
                    if is_speaking:
                        silence_count += 1
                        if silence_count >=35:
                            print("Turn Has Ended")
                            samples = np.frombuffer(buffer, dtype=np.float32)
                            downsampled = samples[::3]
                            segments, info = model.transcribe(downsampled, language="hi")
                            for segment in segments:
                                print("YOU SAID :", segment.text)
                                transcribed_text += segment.text
                            response = client.chat.completions.create(
                                model="openai/gpt-oss-20b",
                                messages=[
                                    {"role": "system", "content": "You are a voice assistant. Reply in plain conversational text with NO markdown, NO asterisks, NO bullet points, NO emojis, and no special symbols. Keep replies short, 1-2 sentences, since they will be read aloud."},
                                    {"role": "user", "content":transcribed_text}
                                ],
                                stream = True,
                            )

                            reply =""
                            sentence_buffer =""

                            for chunk in response:
                                token = chunk.choices[0].delta.content
                                if token:
                                    reply += token
                                    sentence_buffer += token
                                    if any(p in  token for p in ".!?"):
                                        print("SENTENCE:", sentence_buffer)
                                        sentence_buffer = ""
                            print()    
                                                

                              

                            audio_chunks = []
                            for audio_chunk in piper_voice.synthesize(reply):
                                audio_chunks.append(audio_chunk.audio_int16_bytes)
                            audio_bytes = b"".join(audio_chunks)

                            audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
                            audio_float32 = audio_int16.astype(np.float32)/32768.0

                            await websocket.send_bytes(audio_float32.tobytes())

                            buffer = b""
                            is_speaking = False
                            silence_count = 0   
                            transcribed_text = ""   

      
            
    except WebSocketDisconnect:
        pass