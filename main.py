from typing import List, Optional, Union, Dict, Any
import time
import uuid
import random
import string
import sys
import json
import logging

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TARGET_URL = "https://fal.run/fal-ai/any-llm"
STREAM_TARGET_URL = "https://fal.run/fal-ai/any-llm/stream"
VISION_TARGET_URL = "https://fal.run/fal-ai/any-llm/vision"
VISION_STREAM_TARGET_URL = "https://fal.run/fal-ai/any-llm/vision/stream"
ALLOWED_MODEL = "anthropic/claude-3.5-sonnet"
FAL_KEY = "a5411de0-36c9-4e9a-bd61-51da82c9a742:cbc4cf9924de5a5c864b1dbf2b1bf1f0"

class OpenAIError(BaseModel):
    message: str
    type: str
    param: Optional[str] = None
    code: str

class ErrorResponse(BaseModel):
    error: OpenAIError

class ImageUrl(BaseModel):
    url: str
    detail: Optional[str] = None

class TextContent(BaseModel):
    type: str = "text"
    text: str

class ImageContent(BaseModel):
    type: str = "image_url"
    image_url: Union[ImageUrl, str]

class OpenAIMessage(BaseModel):
    role: str
    content: Union[str, List[Union[TextContent, ImageContent]]]

class OpenAIChoice(BaseModel):
    index: int
    message: OpenAIMessage
    finish_reason: Optional[str] = None
    logprobs: Optional[dict] = None

class OpenAIUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class OpenAIResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{''.join(random.choice(string.ascii_letters + string.digits) for _ in range(24))}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[OpenAIChoice]
    usage: OpenAIUsage = Field(default_factory=OpenAIUsage)
    system_fingerprint: str = Field(default_factory=lambda: f"fp_{uuid.uuid4().hex[:10]}")

class OpenAIRequest(BaseModel):
    model: str
    messages: List[OpenAIMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = None
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    response_format: Optional[Dict[str, str]] = None
    seed: Optional[int] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, str]]] = None

async def fal_request_sync(prompt: str, model: str, system_prompt: str, image_url: Optional[str] = None):
    target_url = VISION_TARGET_URL if image_url else TARGET_URL
    
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Key {FAL_KEY}"
    }

    data = {
        "prompt": prompt,
        "model": model,
        "system_prompt": system_prompt,
    }

    if image_url:
        data["image_url"] = image_url

    logger.debug(f"Request to fal (sync): {data}")

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(target_url, headers=headers, json=data, timeout=120.0)
            response.raise_for_status()
            response_data = response.json()
            
            if response_data.get("error"):
                raise HTTPException(status_code=500, detail=response_data["error"])
                
            return response_data
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error during fal request: {e}, {e.response.text}")
            raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
        except Exception as e:
            logger.error(f"Error during fal request: {e}")
            raise HTTPException(status_code=500, detail=str(e))

async def fal_request_stream(prompt: str, model: str, system_prompt: str, image_url: Optional[str] = None):
    target_url = VISION_STREAM_TARGET_URL if image_url else STREAM_TARGET_URL
    
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Key {FAL_KEY}"
    }

    data = {
        "prompt": prompt,
        "model": model,
        "system_prompt": system_prompt,
    }

    if image_url:
        data["image_url"] = image_url

    logger.debug(f"Request to fal (streaming): {data}")

    last_output = ""
    response_id = f"chatcmpl-{''.join(random.choice(string.ascii_letters + string.digits) for _ in range(24))}"
    system_fp = f"fp_{uuid.uuid4().hex[:10]}"

    async with httpx.AsyncClient() as client:
        try:
            async with client.stream("POST", target_url, headers=headers, json=data, timeout=120.0) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data:"):
                        try:
                            data = json.loads(line[5:].strip())
                            if data.get("error"):
                                error_response = {
                                    "id": response_id,
                                    "object": "chat.completion.chunk",
                                    "created": int(time.time()),
                                    "model": model,
                                    "choices": [{
                                        "index": 0,
                                        "message": {"role": "assistant", "content": str(data["error"])},
                                        "finish_reason": "error",
                                        "logprobs": None
                                    }],
                                    "system_fingerprint": system_fp
                                }
                                yield f"data: {json.dumps(error_response)}\n\n"
                                break

                            new_output = data["output"][len(last_output):]
                            if new_output:
                                chunk_response = {
                                    "id": response_id,
                                    "object": "chat.completion.chunk",
                                    "created": int(time.time()),
                                    "model": model,
                                    "choices": [{
                                        "index": 0,
                                        "message": {"role": "assistant", "content": new_output},
                                        "finish_reason": None if data["partial"] else "stop",
                                        "logprobs": None
                                    }],
                                    "system_fingerprint": system_fp
                                }
                                yield f"data: {json.dumps(chunk_response)}\n\n"
                                last_output = data["output"]

                            if not data["partial"]:
                                break
                        except json.JSONDecodeError:
                            continue
                yield "data: [DONE]\n\n"
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error during fal request: {e}, {e.response.text}")
            raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
        except Exception as e:
            logger.error(f"Error during fal request: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "message": "Invalid request format",
                "type": "invalid_request_error",
                "param": None,
                "code": "invalid_json"
            }
        }
    )

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": ALLOWED_MODEL,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "openai"
            }
        ]
    }

@app.post("/v1/chat/completions", response_model=OpenAIResponse)
async def chat_completions(request_body: OpenAIRequest):
    if request_body.model != ALLOWED_MODEL:
        raise HTTPException(
            status_code=400,
            detail="The model does not exist"
        )
    
    logger.debug("=== INCOMING MESSAGES ===")
    for msg in request_body.messages:
        logger.debug(f"Message - Role: {msg.role}, Content: {msg.content}")
    logger.debug("=======================")

    system_message = next((msg for msg in request_body.messages if msg.role == "system"), None)
    system_prompt = system_message.content if system_message else "You are a helpful assistant."

    chat_history = []
    # Process all messages except the last one
    for msg in request_body.messages[:-1]:
        if msg.role == "system":
            continue
        
        if isinstance(msg.content, list):
            text_content = next((c.text for c in msg.content if isinstance(c, TextContent)), "")
            chat_history.append(f"{'Human' if msg.role == 'user' else 'Assistant'}: {text_content}")
        else:
            chat_history.append(f"{'Human' if msg.role == 'user' else 'Assistant'}: {msg.content}")

    last_message = request_body.messages[-1]
    image_url = None
    current_input = ""

    if isinstance(last_message.content, list):
        for content_part in last_message.content:
            if isinstance(content_part, ImageContent):
                if isinstance(content_part.image_url, str):
                    image_url = content_part.image_url
                else:
                    image_url = content_part.image_url.url
            elif isinstance(content_part, TextContent):
                current_input = content_part.text
    else:
        current_input = last_message.content

    formatted_prompt = f"{system_prompt}\n\n"
    if chat_history:
        formatted_prompt += "\n".join(chat_history) + "\n"
    formatted_prompt += f"Human: {current_input}"

    logger.debug("=== FINAL PROMPT ===")
    logger.debug(formatted_prompt)
    logger.debug("===================")

    if not request_body.stream:
        response_data = await fal_request_sync(formatted_prompt, request_body.model, system_prompt, image_url=image_url)
        return OpenAIResponse(
            model=request_body.model,
            choices=[
                OpenAIChoice(
                    index=0,
                    message=OpenAIMessage(
                        role="assistant",
                        content=response_data["output"]
                    ),
                    finish_reason="stop",
                    logprobs=None
                )
            ]
        )

    return StreamingResponse(
        fal_request_stream(formatted_prompt, request_body.model, system_prompt, image_url),
        media_type="text/event-stream",
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
