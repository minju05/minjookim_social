from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

from muma_config import Settings


@dataclass(frozen=True)
class QuestionRecord:
    question_id: str
    episode_id: str
    question_type: str
    question: str
    choices: list[str]
    answer: str
    text_context: str | None
    video_path: Path
    raw: dict[str, Any]


def ensure_dataset(settings: Settings, allow_patterns: list[str] | None = None) -> Path:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.cache_dir.mkdir(parents=True, exist_ok=True)

    local_dir = settings.data_dir / "muma_tom_benchmark"
    if (local_dir / "questions.json").exists():
        return local_dir

    snapshot_download(
        repo_id=settings.hf_repo_id,
        repo_type="dataset",
        token=settings.hf_token,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        cache_dir=settings.cache_dir,
        allow_patterns=allow_patterns,
    )
    return local_dir


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


CHOICE_RE = re.compile(r"^[A-C]\)\s*(.*)$")


def _normalize_choice(choice: Any) -> str:
    if isinstance(choice, str):
        return choice.strip()
    if isinstance(choice, dict):
        for key in ("text", "option", "label", "answer"):
            value = choice.get(key)
            if isinstance(value, str):
                return value.strip()
    return str(choice).strip()


def _detect_question_type(raw: dict[str, Any]) -> str:
    for key in ("question_type", "type", "category", "concept"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "unknown"


def _extract_answer(raw: dict[str, Any], choices: list[str]) -> str:
    answer = raw.get("answer")
    if isinstance(answer, str):
        answer = answer.strip()
        if len(answer) == 1 and answer.upper() in {"A", "B", "C"}:
            index = ord(answer.upper()) - ord("A")
            if 0 <= index < len(choices):
                return choices[index]
        return answer

    answer_idx = raw.get("answer_idx", raw.get("answer_id", raw.get("label")))
    if isinstance(answer_idx, int) and 0 <= answer_idx < len(choices):
        return choices[answer_idx]

    raise ValueError(f"Could not parse answer for question: {raw}")


def _extract_question_id(raw: dict[str, Any], index: int) -> str:
    for key in ("question_id", "id", "qid", "uid"):
        value = raw.get(key)
        if isinstance(value, (str, int)):
            return str(value)
    return f"question_{index}"


def _extract_episode_id(raw: dict[str, Any]) -> str:
    for key in ("episode_id", "video_id", "scene_id", "event_id"):
        value = raw.get(key)
        if isinstance(value, (str, int)):
            return str(value)
    raise ValueError(f"Could not parse episode id from: {raw}")


def _clean_description(text: str | None) -> str | None:
    if not isinstance(text, str):
        return None
    text = text.strip()
    if text.startswith("b'") and text.endswith("'"):
        text = text[2:-1]
    return text.replace("\\n", "\n").strip()


def _parse_compound_question(text: str) -> tuple[str, list[str]]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    question_lines: list[str] = []
    choices: list[str] = []

    for line in lines:
        match = CHOICE_RE.match(line)
        if match:
            choices.append(match.group(1).strip())
        else:
            question_lines.append(line)

    if len(choices) != 3:
        raise ValueError(f"Expected 3 choices, got {len(choices)} from: {text}")
    return " ".join(question_lines).strip(), choices


def _normalize_answer_text(answer: str) -> str:
    answer = answer.strip()
    match = CHOICE_RE.match(answer)
    if match:
        return match.group(1).strip()
    return answer


def load_questions(dataset_dir: Path) -> list[QuestionRecord]:
    return load_questions_from_paths(
        dataset_dir=dataset_dir,
        questions_path=dataset_dir / "questions.json",
        texts_path=dataset_dir / "texts.json",
    )


def load_questions_from_paths(
    dataset_dir: Path,
    questions_path: Path,
    texts_path: Path,
) -> list[QuestionRecord]:
    questions = load_json(questions_path)
    texts = load_json(texts_path)

    text_by_episode: dict[str, str] = {}
    if isinstance(texts, dict):
        for key, value in texts.items():
            if isinstance(value, str):
                text_by_episode[str(key)] = value.strip()
            elif isinstance(value, dict):
                text = value.get("text", value.get("context", value.get("description")))
                if isinstance(text, str):
                    text_by_episode[str(key)] = text.strip()

    records: list[QuestionRecord] = []
    if isinstance(questions, dict):
        for episode_id, payload in questions.items():
            if not isinstance(payload, dict):
                continue

            description = _clean_description(payload.get("description"))
            text_context = text_by_episode.get(str(episode_id)) or description
            question_map = payload.get("questions", {})
            answer_map = payload.get("answers", {})
            label_map = payload.get("labels", {})

            for local_qid, compound_question in question_map.items():
                if not isinstance(compound_question, str):
                    continue

                question_text, choices = _parse_compound_question(compound_question)
                answer_text = _normalize_answer_text(answer_map[str(local_qid)])
                video_path = dataset_dir / "videos" / f"video_{episode_id}.mp4"

                records.append(
                    QuestionRecord(
                        question_id=f"{episode_id}_{local_qid}",
                        episode_id=str(episode_id),
                        question_type=str(label_map.get(str(local_qid), "unknown")),
                        question=question_text,
                        choices=choices,
                        answer=answer_text,
                        text_context=text_context,
                        video_path=video_path,
                        raw={
                            "episode_id": episode_id,
                            "local_question_id": local_qid,
                            "description": description,
                            "question": compound_question,
                            "answer": answer_map.get(str(local_qid)),
                            "label": label_map.get(str(local_qid)),
                        },
                    )
                )
    return records
