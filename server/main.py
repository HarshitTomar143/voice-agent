from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel
import numpy as np
from groq import Groq
from dotenv import load_dotenv
from silero_vad import load_silero_vad
import torch
app = FastAPI()
vad_model = load_silero_vad()
print("VAD model loaded!")

model = WhisperModel("base", device="cpu", compute_type="int8")
print("Whisper model loaded!")

load_dotenv()
print("Environment Variables loaded")
client = Groq()


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

            batch_samples = np.frombuffer(data, dtype=np.float32)   # convert THIS batch
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
                            segments, info = model.transcribe(downsampled, language="en")
                            for segment in segments:
                                print("YOU SAID :", segment.text)
                                transcribed_text += segment.text
                            response = client.chat.completions.create(
                                model="openai/gpt-oss-20b",
                                messages=[
                                    {"role": "user", "content":transcribed_text}
                                ]
                            )                    

                            reply = response.choices[0].message.content
                            print("Ai Reply: ", reply)    
                            buffer = b""
                            is_speaking = False
                            silence_count = 0   
                            transcribed_text = ""   

      
            
    except WebSocketDisconnect:
        pass