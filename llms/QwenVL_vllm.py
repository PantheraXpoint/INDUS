import torch
from llms.BaseModel import BaseVideoModel
from lmdeploy.vl.utils import encode_image_base64
from openai import OpenAI
import os
import asyncio

class QwenVL_vllm(BaseVideoModel):
    def __init__(self, model_type="Qwen/Qwen2.5-14B-Instruct-AWQ", tp=1, port=8000):
        """
        Initialize the QwenVL model.

        Args:
            model_type (str): The type or name of the model.
            tp (int): The number of GPUs to use.
        """
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", "dummy"),  # not checked by vLLM
            base_url=f"http://localhost:{port}/v1"            # your vLLM endpoint
        )
        self.model_type = model_type
        self.client = client


    async def _async_request(self, session_input):
        return await asyncio.to_thread(
            self.client.chat.completions.create,
            model=self.model_type,
            messages=session_input,
            temperature=0.5,
            max_tokens=512,
        )

    def batch_generate_response(self, batch_inputs, max_batch_size=64, max_new_tokens=512, temperature=0.5):
        async def run_all():
            tasks = []
            for inputs in batch_inputs:
                if "video" in inputs:
                    imgs = inputs["video"]
                    content = [
                        {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{encode_image_base64(img)}'}}
                        for img in imgs
                    ]
                    content.append({'type': 'text', 'text': inputs["text"]})
                else:
                    content = [{'type': 'text', 'text': inputs["text"]}]

                messages = [{"role": "user", "content": content}]
                tasks.append(self._async_request(messages))

            results = await asyncio.gather(*tasks)
            return [r.choices[0].message.content.strip() for r in results]

        return asyncio.run(run_all())

        
if __name__ == "__main__":
    model = QwenVL_vllm(model_type="Qwen/Qwen2.5-VL-7B-Instruct-AWQ")
    response = model.batch_generate_response([{"text": "What is the weather in Tokyo?"}, {"text": "What is the weather in Tokyo?"}])
    for r in response:
        print(r)
        print("-"*100)