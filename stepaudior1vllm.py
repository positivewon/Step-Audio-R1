      
import base64
import json
import re
import io
import wave
import os
from datetime import datetime
from textwrap import dedent

import requests
from utils import load_audio


class StepAudioR1:

    audio_token_re = re.compile(r'<audio_(\d+)>')

    def __init__(self, api_url, model_name, chat_template=None):
        self.api_url = api_url
        self.model_name = model_name
        # self.chat_template = chat_template or DEFAULT_CHAT_TEMPLATE
        self.log_dir = "request_logs"
        os.makedirs(self.log_dir, exist_ok=True)

    def __call__(self, messages, **kwargs):
        return next(self.stream(messages, **kwargs, stream=False))

    def log_request(self, payload):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = os.path.join(self.log_dir, f"request_{timestamp}.json")

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        return filename

    def stream(self, messages, stream=True, stop=None, **kwargs):
        headers = {"Content-Type": "application/json"}
        payload = kwargs
        payload["messages"] = self.apply_chat_template(messages)
        payload["model"] = self.model_name
        payload["stream"] = stream
        payload["skip_special_tokens"] = False
        # payload["chat_template"] = self.chat_template

        # Set default stop tokens if none provided
        if stop is None:
            stop = ["<|EOT|>"]
        # payload["stop"] = stop
        if (payload["messages"][-1].get("role", None) == "assistant") and (payload["messages"][-1].get("content", None) is None):
            payload["messages"].pop(-1)
            payload["continue_final_message"] = False
            payload["add_generation_prompt"] = True
        elif payload["messages"][-1].get("eot", True):
            payload["continue_final_message"] = False
            payload["add_generation_prompt"] = True
        else:
            payload["continue_final_message"] = True
            payload["add_generation_prompt"] = False

        self.log_request(payload)

        with requests.post(self.api_url, headers=headers, json=payload, stream=stream) as response:
            response.raise_for_status()

            for line in response.iter_lines():
                if line == b'':
                    continue
                try:
                    # Handle SSE format: "data: {...}"
                    line_str = line.decode('utf-8')
                    if stream and line_str.startswith('data: '):
                        line_str = line_str[6:]  # Remove "data: " prefix

                    if line_str == '[DONE]':
                        break

                    # Parse JSON
                    data = json.loads(line_str)
                    choice_data = data['choices'][0]
                    line_content = choice_data['delta'] if stream else choice_data['message']

                    # print(choice_data)

                    # Extract text and audio
                    text = line_content.get('tts_content', {}).get('tts_text', None)
                    text = text if text is not None else line_content.get('content', '')

                    audio = line_content.get('tts_content', {}).get('tts_audio', None)
                    audio = [int(i) for i in StepAudioR1.audio_token_re.findall(audio)] if audio else None

                    yield line_content, text, audio

                except json.JSONDecodeError:
                    # Skip invalid JSON lines silently in streaming
                    continue
                except (KeyError, IndexError):
                    # Skip malformed response chunks
                    continue

    def process_content_item(self, item):
        if item["type"] == "audio":
            audio_tensor = load_audio(item["audio"], target_rate=16000)
            chunks = []
            for i in range(0, audio_tensor.shape[0], 25 * 16000):
                chunk = audio_tensor[i:i + 25 * 16000]
                if len(chunk.numpy()) == 0:
                    continue
                chunk_int16 = (chunk.numpy().clip(-1.0, 1.0) * 32767.0).astype('int16')
                buf = io.BytesIO()
                with wave.open(buf, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(16000)
                    wf.writeframes(chunk_int16.tobytes())
                chunks.append({"type": "input_audio", "input_audio": {"data": base64.b64encode(buf.getvalue()).decode('utf-8'), "format": "wav"}})
            return chunks
        return [item]

    def apply_chat_template(self, messages):
        out = []
        for m in messages:
            if m["role"] == "human" and isinstance(m["content"], list):
                out.append({"role": m["role"], "content": [j for i in m["content"] for j in self.process_content_item(i)]})
            else:
                out.append(m)
        return out
