import torch
from typing import Any, Dict, List, Optional, Tuple, Union

from llms.BaseModel import BaseVideoModel
from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig
from lmdeploy.vl.utils import encode_image_base64


class QwenVL(BaseVideoModel):
    def __init__(self, model_type="Qwen/Qwen2.5-VL-7B-Instruct-AWQ", tp=1):
        """
        Initialize the QwenVL model.

        Args:
            model_type (str): The type or name of the model.
            tp (int): The number of GPUs to use.
        """
        self.pipe = pipeline(
            model_type,
            backend_config=TurbomindEngineConfig(
                session_len=8192 * 4,
                tp=tp,
                cache_max_entry_count=0.3,
            ),
        )

    @staticmethod
    def _extract_usage(resp: Any) -> Optional[Dict[str, Any]]:
        """
        Extract token usage from an LMDeploy pipeline response.

        LMDeploy pipeline responses commonly expose:
          - input_token_len
          - generate_token_len

        We normalize them into:
          {
              "prompt_tokens": ...,
              "completion_tokens": ...,
              "total_tokens": ...
          }

        Fallbacks are included because exact fields can vary by version/model.
        """
        prompt_tokens = None
        completion_tokens = None

        # Preferred LMDeploy fields
        if hasattr(resp, "input_token_len"):
            try:
                prompt_tokens = int(resp.input_token_len)
            except Exception:
                pass

        if hasattr(resp, "generate_token_len"):
            try:
                completion_tokens = int(resp.generate_token_len)
            except Exception:
                pass

        # Fallback: some backends may expose token_ids / history_token_len
        if prompt_tokens is None and hasattr(resp, "history_token_len"):
            try:
                prompt_tokens = int(resp.history_token_len)
            except Exception:
                pass

        if completion_tokens is None and hasattr(resp, "token_ids"):
            try:
                token_ids = getattr(resp, "token_ids")
                if token_ids is not None:
                    completion_tokens = len(token_ids)
            except Exception:
                pass

        if prompt_tokens is None and completion_tokens is None:
            return None

        return {
            "prompt_tokens": prompt_tokens if prompt_tokens is not None else 0,
            "completion_tokens": completion_tokens if completion_tokens is not None else 0,
            "total_tokens": (prompt_tokens if prompt_tokens is not None else 0)
            + (completion_tokens if completion_tokens is not None else 0),
        }

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

    def generate_response(
        self,
        inputs,
        max_new_tokens=512,
        temperature=0.5,
        log_usage: bool = False,
        return_usage: bool = False,
    ) -> Union[str, Tuple[str, Optional[Dict[str, Any]]]]:
        """
        Generate a response based on the inputs.

        Args:
            inputs (dict): Input data containing text
            {
                "text": str,
                "video": list[Image.Image](optional)
            }

        Returns:
            str
            or (str, usage) if return_usage=True
        """
        assert "text" in inputs.keys(), "Please provide a text prompt."
        gen_config = GenerationConfig(
            do_sample=True,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        if "video" in inputs.keys():
            imgs = inputs["video"]
            question = inputs["text"]

            content = []
            for img in imgs:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "max_dynamic_patch": 1,
                        "url": f"data:image/jpeg;base64,{encode_image_base64(img)}",
                    },
                })
            content.append({"type": "text", "text": question})

            messages = [dict(role="user", content=content)]
            response = self.pipe(messages, gen_config=gen_config)
            text_response = response.text
        else:
            response = self.pipe(inputs["text"], gen_config=gen_config)
            text_response = response.text

        usage = self._extract_usage(response)

        if log_usage:
            if usage is None:
                print("[usage] None (LMDeploy response did not expose token counts)")
            else:
                print(
                    f"[usage] prompt={usage.get('prompt_tokens')} "
                    f"completion={usage.get('completion_tokens')} "
                    f"total={usage.get('total_tokens')}"
                )

        if return_usage:
            return text_response, usage
        return text_response

    def batch_generate_response(
        self,
        batch_inputs,
        max_batch_size=64,
        max_new_tokens=512,
        temperature=0.5,
        log_usage: bool = True,
        return_usage: bool = False,
    ) -> Union[List[str], Tuple[List[str], List[Optional[Dict[str, Any]]], Dict[str, int]]]:
        """
        If return_usage=True, returns:
          (texts, per_request_usages, totals)

        per_request_usages[i] is usage dict or None if LMDeploy didn't expose it.
        """
        prompts = []
        responses = []
        gen_config = GenerationConfig(
            do_sample=True,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        if "video" in batch_inputs[0].keys():
            for inputs in batch_inputs:
                imgs = inputs["video"]
                question = inputs["text"]

                content = []
                for img in imgs:
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "max_dynamic_patch": 1,
                            "url": f"data:image/jpeg;base64,{encode_image_base64(img)}",
                        },
                    })
                content.append({"type": "text", "text": question})

                messages = [dict(role="user", content=content)]
                prompts.append(messages)

            for i in range(0, len(prompts), max_batch_size):
                responses.extend(
                    self.pipe(prompts[i:i + max_batch_size], gen_config=gen_config)
                )
        else:
            for inputs in batch_inputs:
                prompts.append(inputs["text"])

            for i in range(0, len(prompts), max_batch_size):
                responses.extend(
                    self.pipe(prompts[i:i + max_batch_size], gen_config=gen_config)
                )

        texts: List[str] = []
        usages: List[Optional[Dict[str, Any]]] = []

        for i, response in enumerate(responses):
            try:
                texts.append(response.text)
            except Exception:
                texts.append("[ERROR] Malformed response (no text)")

            usage = self._extract_usage(response)
            usages.append(usage)

            if log_usage:
                if usage is None:
                    print(f"[usage {i}] None (LMDeploy response did not expose token counts)")
                else:
                    print(
                        f"[usage {i}] prompt={usage.get('prompt_tokens')} "
                        f"completion={usage.get('completion_tokens')} "
                        f"total={usage.get('total_tokens')}"
                    )

        totals = self._sum_usages(usages)

        if log_usage:
            print(
                f"[usage total] prompt={totals['prompt_tokens']} "
                f"completion={totals['completion_tokens']} "
                f"total={totals['total_tokens']}"
            )

        if return_usage:
            return texts, usages, totals
        return texts


if __name__ == "__main__":
    model = QwenVL(model_type="Qwen/Qwen2.5-VL-7B-Instruct-AWQ")

    out = model.batch_generate_response(
        [
            {"text": "What is the weather in Tokyo?"},
            {"text": "What is the weather in Tokyo?"},
        ],
        log_usage=True,
        return_usage=True,
    )

    texts, usages, totals = out

    for t in texts:
        print(t)
        print("-" * 80)

    print("Per-request usage:", usages)
    print("Totals:", totals)