import asyncio
import os
from typing import Any, Dict, List, Optional, Tuple, Union

from llms.BaseModel import BaseVideoModel
from lmdeploy.vl.utils import encode_image_base64
from lmdeploy import GenerationConfig
from openai import OpenAI


class QwenVL_vllm(BaseVideoModel):
    def __init__(self, model_type: str = "Qwen/Qwen2.5-14B-Instruct-AWQ", tp: int = 1, port: int = 8000):
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", "dummy"),
            base_url=f"http://localhost:{port}/v1",
        )
        self.model_type = model_type
        self.client = client

    @staticmethod
    def _extract_usage(resp: Any) -> Optional[Dict[str, Any]]:
        """
        Extract usage from OpenAI-compatible response.
        vLLM usually returns something like:
          resp.usage = { "prompt_tokens":..., "completion_tokens":..., "total_tokens":... }
        """
        usage = getattr(resp, "usage", None)
        if usage is None:
            return None

        # openai python may return a pydantic object; dict(...) works often
        try:
            if isinstance(usage, dict):
                return usage
            return dict(usage)
        except Exception:
            # fallback: best-effort attribute extraction
            out = {}
            for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                if hasattr(usage, k):
                    out[k] = getattr(usage, k)
            return out or None

    @staticmethod
    def _sum_usages(usages: List[Optional[Dict[str, Any]]]) -> Dict[str, int]:
        totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        for u in usages:
            if not u:
                continue
            for k in totals:
                v = u.get(k)
                if isinstance(v, int):
                    totals[k] += v
        return totals

    async def _async_request(
        self,
        session_input,
        temperature: float,
        max_tokens: int,
        gen_config: Optional[GenerationConfig] = None,
    ):
        """
        Internal helper to perform a single OpenAI-compatible chat completion request.

        If a lmdeploy.GenerationConfig is provided, its fields override the
        corresponding scalar arguments where applicable, so that calling code
        can control decoding in a way similar to QwenLM/QwenVL.
        """
        if gen_config is not None:
            # Map common fields from lmdeploy.GenerationConfig to OpenAI/vLLM params.
            try:
                if hasattr(gen_config, "temperature") and gen_config.temperature is not None:
                    temperature = float(gen_config.temperature)
            except Exception:
                pass
            try:
                if hasattr(gen_config, "max_new_tokens") and gen_config.max_new_tokens is not None:
                    max_tokens = int(gen_config.max_new_tokens)
            except Exception:
                pass

        return await asyncio.to_thread(
            self.client.chat.completions.create,
            model=self.model_type,
            messages=session_input,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def batch_generate_response(
        self,
        batch_inputs: List[Dict[str, Any]],
        max_batch_size: int = 64,
        max_new_tokens: int = 512,
        temperature: float = 0.5,
        log_usage: bool = True,
        return_usage: bool = False,
        gen_config: Optional[GenerationConfig] = None,
    ) -> Union[List[str], Tuple[List[str], List[Optional[Dict[str, Any]]], Dict[str, int]]]:
        """
        If return_usage=True, returns:
          (texts, per_request_usages, totals)

        per_request_usages[i] is usage dict or None if server didn't return it.
        """
        async def run_all():
            tasks = []
            for inputs in batch_inputs:
                if "video" in inputs:
                    imgs = inputs["video"]
                    content = [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encode_image_base64(img)}"},
                        }
                        for img in imgs
                    ]
                    content.append({"type": "text", "text": inputs["text"]})
                else:
                    content = [{"type": "text", "text": inputs["text"]}]

                messages = [{"role": "user", "content": content}]
                tasks.append(
                    self._async_request(
                        messages,
                        temperature=temperature,
                        max_tokens=max_new_tokens,
                        gen_config=gen_config,
                    )
                )

            # Don't crash the whole batch if one request fails
            results = await asyncio.gather(*tasks, return_exceptions=True)

            texts: List[str] = []
            usages: List[Optional[Dict[str, Any]]] = []

            for i, r in enumerate(results):
                if isinstance(r, Exception):
                    texts.append(f"[ERROR] {type(r).__name__}: {r}")
                    usages.append(None)
                    if log_usage:
                        print(
                            f"[usage {i}] ERROR (no usage): {r}",
                            file=__import__("sys").stderr,
                        )
                    continue

                # normal response
                try:
                    texts.append(r.choices[0].message.content.strip())
                except Exception:
                    texts.append("[ERROR] Malformed response (no text)")
                u = self._extract_usage(r)
                usages.append(u)

                if log_usage:
                    if u is None:
                        print(f"[usage {i}] None (server did not return usage)")
                    else:
                        print(
                            f"[usage {i}] prompt={u.get('prompt_tokens')} "
                            f"completion={u.get('completion_tokens')} "
                            f"total={u.get('total_tokens')}"
                        )

            totals = self._sum_usages(usages)
            if log_usage:
                print(f"[usage total] prompt={totals['prompt_tokens']} completion={totals['completion_tokens']} total={totals['total_tokens']}")

            return texts, usages, totals

        texts, usages, totals = asyncio.run(run_all())
        if return_usage:
            return texts, usages, totals
        return texts


if __name__ == "__main__":
    model = QwenVL_vllm(model_type="Qwen/Qwen2.5-14B-Instruct-AWQ")
    out = model.batch_generate_response(
        [{"text": "What is the weather in Tokyo?"}, {"text": "What is the weather in Tokyo?"}],
        log_usage=True,
        return_usage=True,
    )
    texts, usages, totals = out
    for t in texts:
        print(t)
        print("-" * 80)
    print("Totals:", totals)