#!/usr/bin/env python3
"""
Qwen3-VL Server with OpenAI-compatible API
Implements the same API interface as HAMSTER server for easy switching
"""

import argparse
import base64
import os
import re
import time
from contextlib import asynccontextmanager
from io import BytesIO
from typing import List, Literal, Optional, Union

import requests
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image as PILImage
from PIL.Image import Image
from pydantic import BaseModel
from transformers import AutoModelForImageTextToText, AutoProcessor


# ============================================================================
# Pydantic Models for OpenAI-compatible API
# ============================================================================

class TextContent(BaseModel):
    type: Literal["text"]
    text: str


class ImageURL(BaseModel):
    url: str


class ImageContent(BaseModel):
    type: Literal["image_url"]
    image_url: ImageURL


IMAGE_CONTENT_BASE64_REGEX = re.compile(r"^data:image/(png|jpe?g);base64,(.*)$")


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: Union[str, List[Union[TextContent, ImageContent]]]


class ChatCompletionRequest(BaseModel):
    model: Literal[
        "Qwen3-VL-8B-Instruct",
        "Qwen3-VL-4B-Instruct",
        "Qwen3-VL-2B-Instruct",
        "Qwen2.5-VL-7B-Instruct",
    ]
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 512
    top_p: Optional[float] = 0.8
    top_k: Optional[int] = 20
    temperature: Optional[float] = 0.7
    repetition_penalty: Optional[float] = 1.0
    presence_penalty: Optional[float] = None  # Alias for repetition_penalty (OpenAI compat)
    stream: Optional[bool] = False


# ============================================================================
# Global Model State
# ============================================================================

model = None
processor = None
model_name = None


def load_image(image_url: str) -> Image:
    """Load image from URL or base64 string"""
    if image_url.startswith("http") or image_url.startswith("https"):
        response = requests.get(image_url)
        image = PILImage.open(BytesIO(response.content)).convert("RGB")
    else:
        match_results = IMAGE_CONTENT_BASE64_REGEX.match(image_url)
        if match_results is None:
            raise ValueError(f"Invalid image url: {image_url}")
        image_base64 = match_results.groups()[1]
        image = PILImage.open(BytesIO(base64.b64decode(image_base64))).convert("RGB")
    return image


# ============================================================================
# FastAPI Application
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup"""
    global model, processor, model_name

    print("=" * 60)
    print("Loading Qwen3-VL model...")
    print(f"Model: {model_name}")
    print("=" * 60)

    # Load model and processor
    # Use device_map="auto" to automatically distribute across available GPUs
    # Use single GPU (controlled by CUDA_VISIBLE_DEVICES)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )

    processor = AutoProcessor.from_pretrained(model_name)

    print("=" * 60)
    print("Model loaded successfully!")
    print(f"Device: {next(model.parameters()).device}")
    print(f"Dtype: {next(model.parameters()).dtype}")
    print("=" * 60)

    yield

    # Cleanup
    print("Shutting down server...")


app = FastAPI(lifespan=lifespan)


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint
    Mimics the HAMSTER server API for seamless switching
    """
    try:
        # Prepare messages for Qwen3-VL
        messages = []
        images = []

        for message in request.messages:
            qwen_message = {
                "role": message.role,
                "content": []
            }

            # Parse content
            if isinstance(message.content, str):
                qwen_message["content"].append({
                    "type": "text",
                    "text": message.content
                })
            elif isinstance(message.content, list):
                for content in message.content:
                    if isinstance(content, TextContent):
                        qwen_message["content"].append({
                            "type": "text",
                            "text": content.text
                        })
                    elif isinstance(content, ImageContent):
                        image = load_image(content.image_url.url)
                        images.append(image)
                        qwen_message["content"].append({
                            "type": "image",
                            "image": image
                        })

            messages.append(qwen_message)

        # Prepare input
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = processor(
            text=[text],
            images=images if images else None,
            return_tensors="pt"
        ).to(model.device)

        # Generate
        # Use presence_penalty as alias for repetition_penalty (OpenAI compatibility)
        rep_penalty = request.repetition_penalty
        if request.presence_penalty is not None:
            rep_penalty = request.presence_penalty

        print(f"Generating response...")
        print(f"  Prompt length: {len(text)} chars")
        print(f"  Images: {len(images)}")
        print(f"  Max tokens: {request.max_tokens}")
        print(f"  Temperature: {request.temperature}, top_p: {request.top_p}, top_k: {request.top_k}")
        print(f"  Repetition penalty: {rep_penalty}")

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature if request.temperature > 0 else None,
                top_p=request.top_p if request.temperature > 0 else None,
                top_k=request.top_k if request.temperature > 0 else None,
                repetition_penalty=rep_penalty if rep_penalty > 1.0 else None,
                do_sample=request.temperature > 0,
            )

        # Decode response
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        print(f"Generated response ({len(output_text)} chars)")

        # Return OpenAI-compatible response
        response = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": output_text
                            }
                        ]
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(inputs.input_ids[0]),
                "completion_tokens": len(generated_ids_trimmed[0]),
                "total_tokens": len(inputs.input_ids[0]) + len(generated_ids_trimmed[0])
            }
        }

        return JSONResponse(content=response)

    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return JSONResponse(
            status_code=500,
            content={"error": error_msg}
        )


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok", "model": model_name}


# ============================================================================
# Main
# ============================================================================

def main():
    global model_name

    parser = argparse.ArgumentParser(description="Qwen3-VL Server with OpenAI-compatible API")
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen3-VL-8B-Instruct",
                       help="Model name or path (default: Qwen/Qwen3-VL-8B-Instruct)")
    parser.add_argument("--port", type=int, default=8001,
                       help="Port number (default: 8001)")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                       help="Host address (default: 0.0.0.0)")

    args = parser.parse_args()
    model_name = args.model_path

    print("\n" + "=" * 60)
    print("Qwen3-VL Server")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Server: http://{args.host}:{args.port}")
    print(f"API Endpoint: http://{args.host}:{args.port}/v1/chat/completions")
    print(f"Health Check: http://{args.host}:{args.port}/health")
    print("=" * 60 + "\n")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )


if __name__ == "__main__":
    main()
