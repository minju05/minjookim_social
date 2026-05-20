from __future__ import annotations

import base64
import json
import random
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any


def _sanitize_json(text: str) -> str:
    """Gemini 응답에서 JSON 파싱을 깨는 제어문자 제거/이스케이프."""
    # 1단계: 명백히 무효한 제어문자 제거
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # 2단계: json.loads로 바로 파싱 가능하면 그대로 반환
    try:
        json.loads(text)
        return text
    except (json.JSONDecodeError, ValueError):
        pass
    # 3단계: literal newline/tab이 JSON string 내부에 있어 invalid한 경우
    # choice_letter를 regex로 직접 추출할 수 있도록 원본 반환
    return text

from google import genai
from google.genai import types

from muma_config import Settings
from muma_data import QuestionRecord
from muma_gpt4o import PROMPT_CONFIGS
from muma_video import EncodedFrame


@dataclass
class GeminiPrediction:
    question_id: str
    episode_id: str
    question_type: str
    gold_answer: str
    predicted_answer: str
    predicted_letter: str
    correct: bool
    reasoning: str
    raw_response: str = ""
    # structured fields (populated when model returns JSON with these keys)
    question_polarity: str = ""
    condition: str = ""
    target_character: str = ""
    key_inference: str = ""
    option_analysis: dict = field(default_factory=dict)


def upload_video_file(client: genai.Client, video_path: str) -> str:
    """mp4 파일을 Gemini File API로 업로드하고 URI를 반환한다.

    업로드 후 PROCESSING → ACTIVE 상태까지 대기.
    """
    import pathlib
    uploaded = client.files.upload(
        file=pathlib.Path(video_path),
        config=types.UploadFileConfig(mime_type="video/mp4"),
    )
    # PROCESSING 상태가 끝날 때까지 폴링
    while uploaded.state and uploaded.state.name == "PROCESSING":
        time.sleep(2)
        uploaded = client.files.get(name=uploaded.name)
    if uploaded.state and uploaded.state.name == "FAILED":
        raise RuntimeError(f"File upload failed: {video_path}")
    return uploaded.uri


def upload_frame_jpeg(client: genai.Client, image_b64: str) -> str:
    """base64 JPEG 프레임을 Gemini File API로 업로드하고 URI를 반환한다.

    base64 inline_data 대신 File API URI를 사용해 request payload를 줄인다.
    """
    import io
    import pathlib
    import tempfile
    jpeg_bytes = base64.b64decode(image_b64)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(jpeg_bytes)
        tmp_path = tmp.name
    try:
        uploaded = client.files.upload(
            file=pathlib.Path(tmp_path),
            config=types.UploadFileConfig(mime_type="image/jpeg"),
        )
    finally:
        import os
        os.unlink(tmp_path)
    return uploaded.uri


def predict_question_frame_files(
    client: genai.Client,
    settings: Settings,
    record: QuestionRecord,
    frame_uris: list[tuple[str, float]],  # (file_uri, second)
    prompt_version: str = "video_only",
    thinking_budget: int = 0,
) -> GeminiPrediction:
    """File API URI 리스트로 stride-20 프레임을 전달해 예측한다.

    GPT-4o와 동일한 stride-20 프레임을 사용하되, base64 대신 File API URI를 사용해
    payload 크기를 줄이고 503 ERR을 방지한다.
    """
    config = PROMPT_CONFIGS[prompt_version]

    parts: list[types.Part] = []
    for uri, second in frame_uris:
        parts.append(types.Part(
            file_data=types.FileData(file_uri=uri, mime_type="image/jpeg")
        ))
        parts.append(types.Part(text=f"Frame timestamp: {second:.1f} seconds"))
    parts.append(types.Part(text=config.payload_fn(record)))

    gen_config_kwargs: dict[str, Any] = {
        "temperature": 0.0,
        "thinking_config": types.ThinkingConfig(thinking_budget=thinking_budget),
    }
    if config.system_prompt:
        gen_config_kwargs["system_instruction"] = config.system_prompt
    if config.json_output:
        gen_config_kwargs["response_mime_type"] = "application/json"

    MAX_RETRIES = 10
    MAX_SERVER_RETRIES = 3
    server_attempts = 0
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=settings.gemini_model,
                contents=[types.Content(parts=parts, role="user")],
                config=types.GenerateContentConfig(**gen_config_kwargs),
            )
            break
        except Exception as e:
            err_str = str(e)
            err_type = type(e).__name__
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                wait = 60 * (attempt + 1)
                print(f"\n[429] Rate limit. {wait}s 대기 후 재시도 ({attempt+1}/{MAX_RETRIES})...")
                time.sleep(wait)
            elif (
                "503" in err_str or "UNAVAILABLE" in err_str
                or "Timeout" in err_type or "timeout" in err_str.lower()
                or "timed out" in err_str.lower()
                or "ConnectError" in err_type or "RemoteProtocol" in err_type
                or "ConnectionError" in err_type
            ):
                server_attempts += 1
                if server_attempts > MAX_SERVER_RETRIES:
                    raise RuntimeError(f"ServerError {MAX_SERVER_RETRIES}회 초과, skip: {record.question_id}")
                wait = 15 * server_attempts
                print(f"\n[{err_type}] 일시적 오류. {wait}s 대기 후 재시도 ({server_attempts}/{MAX_SERVER_RETRIES})...")
                time.sleep(wait)
            else:
                raise
    else:
        raise RuntimeError(f"Max retries ({MAX_RETRIES}) exceeded for question {record.question_id}")

    raw_text = (response.text or "").strip()
    sanitized = _sanitize_json(raw_text)

    try:
        letter = config.response_parser_fn(sanitized)
    except Exception:
        m = re.search(r'"choice_letter"\s*:\s*"([ABCabc])"', sanitized)
        if m:
            letter = m.group(1).upper()
        else:
            return GeminiPrediction(
                question_id=record.question_id,
                episode_id=record.episode_id,
                question_type=record.question_type,
                gold_answer=record.answer,
                predicted_answer="ERR",
                predicted_letter="ERR",
                correct=False,
                reasoning=sanitized,
                raw_response=raw_text,
            )

    if letter not in ("A", "B", "C"):
        return GeminiPrediction(
            question_id=record.question_id,
            episode_id=record.episode_id,
            question_type=record.question_type,
            gold_answer=record.answer,
            predicted_answer="ERR",
            predicted_letter="ERR",
            correct=False,
            reasoning=sanitized,
            raw_response=raw_text,
        )

    index = ord(letter) - ord("A")
    predicted_answer = record.choices[index]

    question_polarity = condition = target_character = key_inference = ""
    option_analysis: dict = {}
    try:
        parsed = json.loads(sanitized)
        question_polarity = parsed.get("question_polarity", "")
        condition = parsed.get("condition", "")
        target_character = parsed.get("target_character", "")
        key_inference = parsed.get("key_inference", "")
        option_analysis = parsed.get("option_analysis", {})
    except (json.JSONDecodeError, AttributeError):
        pass

    return GeminiPrediction(
        question_id=record.question_id,
        episode_id=record.episode_id,
        question_type=record.question_type,
        gold_answer=record.answer,
        predicted_answer=predicted_answer,
        predicted_letter=letter,
        correct=predicted_answer == record.answer,
        reasoning=sanitized,
        raw_response=raw_text,
        question_polarity=question_polarity,
        condition=condition,
        target_character=target_character,
        key_inference=key_inference,
        option_analysis=option_analysis,
    )


def predict_question_video_file(
    client: genai.Client,
    settings: Settings,
    record: QuestionRecord,
    video_uri: str,
    prompt_version: str = "video_only",
    thinking_budget: int = 0,
) -> GeminiPrediction:
    """File API URI를 사용해 mp4 전체를 Gemini에 전달하여 예측한다.

    base64 inline_data 대신 file_data를 사용하므로 payload가 작고 안정적이다.
    """
    config = PROMPT_CONFIGS[prompt_version]

    parts: list[types.Part] = [
        types.Part(
            file_data=types.FileData(file_uri=video_uri, mime_type="video/mp4")
        ),
        types.Part(text=config.payload_fn(record)),
    ]

    gen_config_kwargs: dict[str, Any] = {
        "temperature": 0.0,
        "thinking_config": types.ThinkingConfig(thinking_budget=thinking_budget),
    }
    if config.system_prompt:
        gen_config_kwargs["system_instruction"] = config.system_prompt
    if config.json_output:
        gen_config_kwargs["response_mime_type"] = "application/json"

    MAX_RETRIES = 10
    MAX_SERVER_RETRIES = 3
    server_attempts = 0
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=settings.gemini_model,
                contents=[types.Content(parts=parts, role="user")],
                config=types.GenerateContentConfig(**gen_config_kwargs),
            )
            break
        except Exception as e:
            err_str = str(e)
            err_type = type(e).__name__
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                wait = 60 * (attempt + 1)
                print(f"\n[429] Rate limit. {wait}s 대기 후 재시도 ({attempt+1}/{MAX_RETRIES})...")
                time.sleep(wait)
            elif (
                "503" in err_str or "UNAVAILABLE" in err_str
                or "Timeout" in err_type or "timeout" in err_str.lower()
                or "timed out" in err_str.lower()
                or "ConnectError" in err_type or "RemoteProtocol" in err_type
                or "ConnectionError" in err_type
            ):
                server_attempts += 1
                if server_attempts > MAX_SERVER_RETRIES:
                    raise RuntimeError(f"ServerError {MAX_SERVER_RETRIES}회 초과, skip: {record.question_id}")
                wait = 15 * server_attempts
                print(f"\n[{err_type}] 일시적 오류. {wait}s 대기 후 재시도 ({server_attempts}/{MAX_SERVER_RETRIES})...")
                time.sleep(wait)
            else:
                raise
    else:
        raise RuntimeError(f"Max retries ({MAX_RETRIES}) exceeded for question {record.question_id}")

    raw_text = (response.text or "").strip()
    sanitized = _sanitize_json(raw_text)

    try:
        letter = config.response_parser_fn(sanitized)
    except Exception:
        m = re.search(r'"choice_letter"\s*:\s*"([ABCabc])"', sanitized)
        if m:
            letter = m.group(1).upper()
        else:
            return GeminiPrediction(
                question_id=record.question_id,
                episode_id=record.episode_id,
                question_type=record.question_type,
                gold_answer=record.answer,
                predicted_answer="ERR",
                predicted_letter="ERR",
                correct=False,
                reasoning=sanitized,
                raw_response=raw_text,
            )

    if letter not in ("A", "B", "C"):
        return GeminiPrediction(
            question_id=record.question_id,
            episode_id=record.episode_id,
            question_type=record.question_type,
            gold_answer=record.answer,
            predicted_answer="ERR",
            predicted_letter="ERR",
            correct=False,
            reasoning=sanitized,
            raw_response=raw_text,
        )

    index = ord(letter) - ord("A")
    predicted_answer = record.choices[index]

    question_polarity = condition = target_character = key_inference = ""
    option_analysis: dict = {}
    try:
        parsed = json.loads(sanitized)
        question_polarity = parsed.get("question_polarity", "")
        condition = parsed.get("condition", "")
        target_character = parsed.get("target_character", "")
        key_inference = parsed.get("key_inference", "")
        option_analysis = parsed.get("option_analysis", {})
    except (json.JSONDecodeError, AttributeError):
        pass

    return GeminiPrediction(
        question_id=record.question_id,
        episode_id=record.episode_id,
        question_type=record.question_type,
        gold_answer=record.answer,
        predicted_answer=predicted_answer,
        predicted_letter=letter,
        correct=predicted_answer == record.answer,
        reasoning=sanitized,
        raw_response=raw_text,
        question_polarity=question_polarity,
        condition=condition,
        target_character=target_character,
        key_inference=key_inference,
        option_analysis=option_analysis,
    )


def build_client(settings: Settings, timeout: int = 60) -> genai.Client:
    if not settings.gemini_api_key:
        raise ValueError("GEMINI_API_KEY is not set.")
    http_options = types.HttpOptions(timeout=timeout * 1000)  # ms 단위
    return genai.Client(api_key=settings.gemini_api_key, http_options=http_options)


def predict_question(
    client: genai.Client,
    settings: Settings,
    record: QuestionRecord,
    frames: list[EncodedFrame],
    prompt_version: str = "text_only",
    thinking_budget: int = 0,
) -> GeminiPrediction:
    """
    GPT-4o와 동일한 PROMPT_CONFIGS를 사용해 Gemini로 예측.

    Args:
        frames: sample_video_frames()로 얻은 프레임 목록.
                text_only 실험은 빈 리스트([])를 전달.
        thinking_budget: 0이면 thinking 비활성화 (비용 절감).
    """
    config = PROMPT_CONFIGS[prompt_version]

    # 1. 프레임 → Gemini inline image parts (timestamp 텍스트 포함)
    parts: list[types.Part] = []
    for frame in frames:
        parts.append(types.Part(
            inline_data=types.Blob(
                mime_type="image/jpeg",
                data=base64.b64decode(frame.image_b64),
            )
        ))
        parts.append(types.Part(text=f"Frame timestamp: {frame.second:.1f} seconds"))

    # 2. 텍스트 프롬프트 추가 (GPT-4o payload_fn 그대로 사용)
    parts.append(types.Part(text=config.payload_fn(record)))

    # 3. GenerateContentConfig 구성
    gen_config_kwargs: dict[str, Any] = {
        "temperature": 0.0,
        "thinking_config": types.ThinkingConfig(thinking_budget=thinking_budget),
    }
    if config.system_prompt:
        gen_config_kwargs["system_instruction"] = config.system_prompt
    if config.json_output:
        gen_config_kwargs["response_mime_type"] = "application/json"

    # 4. API 호출 (429/503 재시도 포함)
    MAX_RETRIES = 10
    MAX_SERVER_RETRIES = 2  # 503/timeout은 2회만 재시도 후 ERR 처리
    server_attempts = 0
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=settings.gemini_model,
                contents=[types.Content(parts=parts, role="user")],
                config=types.GenerateContentConfig(**gen_config_kwargs),
            )
            break
        except Exception as e:
            err_str = str(e)
            err_type = type(e).__name__
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                wait = 60 * (attempt + 1)
                print(f"\n[429] Rate limit. {wait}s 대기 후 재시도 ({attempt+1}/{MAX_RETRIES})...")
                time.sleep(wait)
            elif (
                "503" in err_str or "UNAVAILABLE" in err_str
                or "Timeout" in err_type or "timeout" in err_str.lower()
                or "timed out" in err_str.lower()
                or "ConnectError" in err_type or "RemoteProtocol" in err_type
                or "ConnectionError" in err_type
            ):
                server_attempts += 1
                if server_attempts > MAX_SERVER_RETRIES:
                    raise RuntimeError(f"ServerError {MAX_SERVER_RETRIES}회 초과, skip: {record.question_id}")
                wait = 15 * server_attempts
                print(f"\n[{err_type}] 일시적 오류. {wait}s 대기 후 재시도 ({server_attempts}/{MAX_SERVER_RETRIES})...")
                time.sleep(wait)
            else:
                raise
    else:
        raise RuntimeError(f"Max retries ({MAX_RETRIES}) exceeded for question {record.question_id}")

    raw_text = (response.text or "").strip()
    sanitized = _sanitize_json(raw_text)

    # 5. 응답 파싱 (GPT-4o와 동일 parser 사용)
    # 파싱 실패 시 regex fallback → 그래도 실패하면 ERR 마커
    try:
        letter = config.response_parser_fn(sanitized)
    except Exception:
        # regex fallback: "choice_letter": "A" 패턴 직접 추출
        m = re.search(r'"choice_letter"\s*:\s*"([ABCabc])"', sanitized)
        if m:
            letter = m.group(1).upper()
        else:
            return GeminiPrediction(
                question_id=record.question_id,
                episode_id=record.episode_id,
                question_type=record.question_type,
                gold_answer=record.answer,
                predicted_answer="ERR",
                predicted_letter="ERR",
                correct=False,
                reasoning=sanitized,
                raw_response=raw_text,
            )

    index = ord(letter) - ord("A")
    predicted_answer = record.choices[index]

    # 6. structured fields 추출 (있는 경우)
    question_polarity = condition = target_character = key_inference = ""
    option_analysis: dict = {}

    try:
        parsed = json.loads(sanitized)
        question_polarity = parsed.get("question_polarity", "")
        condition = parsed.get("condition", "")
        target_character = parsed.get("target_character", "")
        key_inference = parsed.get("key_inference", "")
        option_analysis = parsed.get("option_analysis", {})
    except (json.JSONDecodeError, AttributeError):
        pass

    return GeminiPrediction(
        question_id=record.question_id,
        episode_id=record.episode_id,
        question_type=record.question_type,
        gold_answer=record.answer,
        predicted_answer=predicted_answer,
        predicted_letter=letter,
        correct=predicted_answer == record.answer,
        reasoning=sanitized,  # GPT-4o와 동일: raw JSON 텍스트 저장
        raw_response=raw_text,
        question_polarity=question_polarity,
        condition=condition,
        target_character=target_character,
        key_inference=key_inference,
        option_analysis=option_analysis,
    )


def prediction_to_dict(prediction: GeminiPrediction) -> dict[str, Any]:
    return asdict(prediction)
