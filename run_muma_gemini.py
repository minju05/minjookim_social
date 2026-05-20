from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from muma_config import load_settings
from muma_data import ensure_dataset, load_questions
from muma_gemini import GeminiPrediction, build_client, prediction_to_dict, predict_question, predict_question_frame_files, predict_question_video_file, upload_frame_jpeg, upload_video_file
from muma_video import sample_video_frames


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a Gemini MuMA-ToM baseline.")
    parser.add_argument("--limit", type=int, default=10, help="Max number of questions to run.")
    parser.add_argument("--offset", type=int, default=0, help="Question offset.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/muma_gemini_predictions.jsonl"),
        help="Output jsonl path.",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download the dataset and print a short summary.",
    )
    parser.add_argument("--frame-stride", type=int, default=None, help="Override frame stride.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Override max frames per question. 0 = no cap.",
    )
    parser.add_argument(
        "--prompt-version",
        type=str,
        default="text_only",
        help="Prompt version key (e.g. text_only, video_only).",
    )
    parser.add_argument(
        "--no-frames",
        action="store_true",
        help="Skip frame sampling and send text only.",
    )
    parser.add_argument(
        "--question-types",
        type=str,
        default=None,
        help="콤마 구분 question_type 필터 (예: belief,social_goal). 미지정 시 전체.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="기존 output 파일에 이어서 쓰기.",
    )
    parser.add_argument(
        "--rerun-err-from",
        type=Path,
        default=None,
        help="기존 jsonl에서 ERR 항목만 재실행 후 in-place 교체 (--output은 대상 파일과 같아야 함).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override Gemini model name (e.g. gemini-2.5-flash).",
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=0,
        help="Thinking token budget. 0 = thinking 비활성화 (기본값, 비용 절감).",
    )
    parser.add_argument(
        "--video-file-api",
        action="store_true",
        help="mp4를 File API로 업로드해서 전달 (base64 inline_data 대신). cond4 권장.",
    )
    parser.add_argument(
        "--frame-uri-cache",
        type=Path,
        default=None,
        help="프레임 URI 캐시 JSON 경로. 지정 시: 파일 없으면 업로드 후 저장, 있으면 로드해서 재사용 (업로드 스킵).",
    )
    parser.add_argument(
        "--video-mp4-api",
        action="store_true",
        help="mp4 파일 전체를 File API로 업로드해서 전달. JPEG 프레임 방식보다 빠름 (에피소드당 1회 업로드).",
    )
    parser.add_argument(
        "--video-uri-cache",
        type=Path,
        default=None,
        help="mp4 URI 캐시 JSON 경로. 지정 시: 없으면 업로드 후 저장, 있으면 로드해서 재사용.",
    )
    return parser


def summarize(records, predictions) -> dict[str, object]:
    by_type_total = defaultdict(int)
    by_type_correct = defaultdict(int)

    for record, prediction in zip(records, predictions):
        by_type_total[record.question_type] += 1
        if prediction.correct:
            by_type_correct[record.question_type] += 1

    summary = {
        "num_questions": len(predictions),
        "overall_accuracy": (sum(p.correct for p in predictions) / len(predictions)) if predictions else 0.0,
        "accuracy_by_type": {},
    }
    for question_type, total in sorted(by_type_total.items()):
        summary["accuracy_by_type"][question_type] = by_type_correct[question_type] / total
    return summary


def main() -> None:
    args = build_arg_parser().parse_args()
    settings = load_settings()
    if args.model:
        import dataclasses
        settings = dataclasses.replace(settings, gemini_model=args.model)

    dataset_dir = ensure_dataset(settings)
    questions = load_questions(dataset_dir)

    if args.download_only:
        print(json.dumps({"dataset_dir": str(dataset_dir), "num_questions": len(questions)}, indent=2))
        return

    # --rerun-err-from: ERR 항목만 재실행 후 in-place 교체
    if args.rerun_err_from:
        src = args.rerun_err_from
        existing = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
        err_ids = {d["question_id"] for d in existing if d.get("predicted_letter") == "ERR"}
        print(f"ERR 항목 {len(err_ids)}개 재실행: {sorted(err_ids)[:5]}...")

        client = build_client(settings, timeout=120 if not args.no_frames else 60)
        frame_stride = args.frame_stride or settings.frame_stride
        max_frames = settings.max_frames if args.max_frames is None else (None if args.max_frames == 0 else args.max_frames)

        out_path = args.output if args.output != Path("outputs/muma_gemini_predictions.jsonl") else src

        q_by_id = {q.question_id: q for q in questions}

        # --video-mp4-api: ERR 질문의 mp4 업로드 (또는 캐시 로드)
        rerun_video_uri_cache: dict[str, str] = {}
        if args.video_mp4_api:
            cache_path = args.video_uri_cache
            if cache_path and cache_path.exists():
                print(f"[mp4 API] 캐시 로드: {cache_path} (업로드 스킵)")
                rerun_video_uri_cache = json.loads(cache_path.read_text(encoding="utf-8"))
                print(f"[mp4 API] 캐시 로드 완료: {len(rerun_video_uri_cache)}개 에피소드")
            else:
                err_records = [q_by_id[qid] for qid in err_ids if qid in q_by_id]
                unique_eps = {r.episode_id: r.video_path for r in err_records}
                print(f"[mp4 API] ERR 에피소드 {len(unique_eps)}개 mp4 업로드 중...")
                for ep_id, vpath in tqdm(unique_eps.items(), desc="Upload mp4"):
                    try:
                        uri = upload_video_file(client, vpath)
                        rerun_video_uri_cache[ep_id] = uri
                    except Exception as e:
                        print(f"  [WARN] {ep_id} mp4 업로드 실패: {e}")
                print(f"[mp4 API] 업로드 완료: {len(rerun_video_uri_cache)}개 에피소드")
                if cache_path:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_path.write_text(
                        json.dumps(rerun_video_uri_cache, ensure_ascii=False, indent=2), encoding="utf-8"
                    )
                    print(f"[mp4 API] 캐시 저장: {cache_path}")

        # --video-file-api: ERR 질문의 에피소드만 골라서 업로드 (또는 캐시 로드)
        rerun_frame_uri_cache: dict[str, list[tuple[str, float]]] = {}
        if args.video_file_api:
            cache_path = args.frame_uri_cache
            if cache_path and cache_path.exists():
                print(f"[File API] 캐시 로드: {cache_path} (업로드 스킵)")
                raw = json.loads(cache_path.read_text(encoding="utf-8"))
                rerun_frame_uri_cache = {ep: [tuple(x) for x in uris] for ep, uris in raw.items()}
                total_frames = sum(len(v) for v in rerun_frame_uri_cache.values())
                print(f"[File API] 캐시 로드 완료: {len(rerun_frame_uri_cache)}개 에피소드, {total_frames}개 프레임")
            else:
                err_records = [q_by_id[qid] for qid in err_ids if qid in q_by_id]
                unique_eps = {r.episode_id: r.video_path for r in err_records}
                print(f"[File API] ERR 에피소드 {len(unique_eps)}개 프레임 업로드 중 (stride={frame_stride})...")
                for ep_id, vpath in tqdm(unique_eps.items(), desc="Upload frames"):
                    frames = sample_video_frames(vpath, frame_stride=frame_stride, max_frames=max_frames)
                    uris = []
                    for fr in frames:
                        try:
                            uri = upload_frame_jpeg(client, fr.image_b64)
                            uris.append((uri, fr.second))
                        except Exception as e:
                            print(f"  [WARN] {ep_id} frame@{fr.second:.1f}s 업로드 실패: {e}")
                    rerun_frame_uri_cache[ep_id] = uris
                total_frames = sum(len(v) for v in rerun_frame_uri_cache.values())
                print(f"[File API] 업로드 완료: {total_frames}개 프레임")
                if cache_path:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_path.write_text(
                        json.dumps(rerun_frame_uri_cache, ensure_ascii=False, indent=2), encoding="utf-8"
                    )
                    print(f"[File API] 캐시 저장: {cache_path}")

        new_results: dict[str, dict] = {}
        for qid in tqdm(sorted(err_ids), desc="Rerun ERR"):
            record = q_by_id.get(qid)
            if record is None:
                print(f"  [WARN] {qid} not found in dataset")
                continue
            try:
                if args.video_mp4_api:
                    video_uri = rerun_video_uri_cache.get(record.episode_id)
                    if not video_uri:
                        raise RuntimeError(f"mp4 URI 없음: episode {record.episode_id}")
                    prediction = predict_question_video_file(
                        client, settings, record, video_uri,
                        prompt_version=args.prompt_version,
                        thinking_budget=args.thinking_budget,
                    )
                elif args.video_file_api:
                    frame_uris = rerun_frame_uri_cache.get(record.episode_id, [])
                    if not frame_uris:
                        raise RuntimeError(f"프레임 URI 없음: episode {record.episode_id}")
                    prediction = predict_question_frame_files(
                        client, settings, record, frame_uris,
                        prompt_version=args.prompt_version,
                        thinking_budget=args.thinking_budget,
                    )
                else:
                    frames = [] if args.no_frames else sample_video_frames(
                        record.video_path, frame_stride=frame_stride, max_frames=max_frames
                    )
                    prediction = predict_question(
                        client, settings, record, frames,
                        prompt_version=args.prompt_version,
                        thinking_budget=args.thinking_budget,
                    )
            except Exception as e:
                print(f"\n[ERROR] {qid}: {e}")
                prediction = GeminiPrediction(
                    question_id=record.question_id,
                    episode_id=record.episode_id,
                    question_type=record.question_type,
                    gold_answer=record.answer,
                    predicted_answer="ERR",
                    predicted_letter="ERR",
                    correct=False,
                    reasoning=f"ERROR: {e}",
                    raw_response="",
                )
            new_results[qid] = prediction_to_dict(prediction)

            # 중간 저장: 처리된 항목 즉시 반영
            merged_intermediate = []
            for d in existing:
                if d["question_id"] in new_results:
                    merged_intermediate.append(new_results[d["question_id"]])
                else:
                    merged_intermediate.append(d)
            out_path.write_text(
                "\n".join(json.dumps(d, ensure_ascii=False) for d in merged_intermediate) + "\n",
                encoding="utf-8",
            )

        # 최종 병합 결과
        merged = []
        for d in existing:
            if d["question_id"] in new_results:
                merged.append(new_results[d["question_id"]])
            else:
                merged.append(d)

        out_path = args.output if args.output != Path("outputs/muma_gemini_predictions.jsonl") else src
        out_path.write_text(
            "\n".join(json.dumps(d, ensure_ascii=False) for d in merged) + "\n",
            encoding="utf-8",
        )
        print(f"\nMerge 완료 → {out_path} ({len(merged)}줄)")
        # 요약
        from collections import defaultdict
        by_type: dict = defaultdict(lambda: {"total": 0, "correct": 0, "err": 0})
        for d in merged:
            qt = d["question_type"]
            by_type[qt]["total"] += 1
            if d.get("predicted_letter") == "ERR":
                by_type[qt]["err"] += 1
            elif d.get("correct"):
                by_type[qt]["correct"] += 1
        total_n = len(merged)
        total_c = sum(v["correct"] for v in by_type.values())
        total_e = sum(v["err"] for v in by_type.values())
        print(f"n={total_n}, correct={total_c}, ERR={total_e}, overall={total_c/total_n:.1%}")
        for qt, v in sorted(by_type.items()):
            print(f"  {qt}: {v['correct']/v['total']:.1%} ({v['correct']}/{v['total']}) err={v['err']}")
        return

    # --question-types 필터
    if args.question_types:
        allowed = {t.strip() for t in args.question_types.split(",")}
        questions = [q for q in questions if q.question_type in allowed]

    selected = questions[args.offset : args.offset + args.limit]
    client = build_client(settings, timeout=120 if not args.no_frames else 60)
    frame_stride = args.frame_stride or settings.frame_stride
    max_frames = settings.max_frames if args.max_frames is None else (None if args.max_frames == 0 else args.max_frames)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    file_mode = "a" if args.append else "w"

    # --video-file-api: stride-20 프레임을 JPEG File API로 업로드 후 URI 캐시
    # episode_id -> [(uri, second), ...] 형태로 캐싱
    frame_uri_cache: dict[str, list[tuple[str, float]]] = {}
    if args.video_file_api:
        cache_path = args.frame_uri_cache
        if cache_path and cache_path.exists():
            print(f"[File API] 캐시 로드: {cache_path} (업로드 스킵)")
            raw = json.loads(cache_path.read_text(encoding="utf-8"))
            frame_uri_cache = {ep: [tuple(x) for x in uris] for ep, uris in raw.items()}
            total_frames = sum(len(v) for v in frame_uri_cache.values())
            print(f"[File API] 캐시 로드 완료: {len(frame_uri_cache)}개 에피소드, {total_frames}개 프레임")
        else:
            unique_episodes = {r.episode_id: r.video_path for r in selected}
            print(f"[File API] {len(unique_episodes)}개 에피소드 프레임 업로드 중 (stride={frame_stride})...")
            for ep_id, vpath in tqdm(unique_episodes.items(), desc="Upload frames"):
                frames = sample_video_frames(vpath, frame_stride=frame_stride, max_frames=max_frames)
                uris = []
                for fr in frames:
                    try:
                        uri = upload_frame_jpeg(client, fr.image_b64)
                        uris.append((uri, fr.second))
                    except Exception as e:
                        print(f"  [WARN] {ep_id} frame@{fr.second:.1f}s 업로드 실패: {e}")
                frame_uri_cache[ep_id] = uris
            total_frames = sum(len(v) for v in frame_uri_cache.values())
            print(f"[File API] 업로드 완료: {total_frames}개 프레임")
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(
                    json.dumps(frame_uri_cache, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                print(f"[File API] 캐시 저장: {cache_path}")

    predictions = []
    with args.output.open(file_mode, encoding="utf-8") as f:
        for record in tqdm(selected, desc=f"Gemini {args.prompt_version}"):
            try:
                if args.video_file_api:
                    frame_uris = frame_uri_cache.get(record.episode_id, [])
                    if not frame_uris:
                        raise RuntimeError(f"프레임 URI 없음: episode {record.episode_id}")
                    prediction = predict_question_frame_files(
                        client, settings, record, frame_uris,
                        prompt_version=args.prompt_version,
                        thinking_budget=args.thinking_budget,
                    )
                else:
                    if args.no_frames:
                        frames = []
                    else:
                        frames = sample_video_frames(
                            record.video_path,
                            frame_stride=frame_stride,
                            max_frames=max_frames,
                        )
                    prediction = predict_question(
                        client, settings, record, frames,
                        prompt_version=args.prompt_version,
                        thinking_budget=args.thinking_budget,
                    )
            except Exception as e:
                print(f"\n[ERROR] {record.question_id}: {e}")
                prediction = GeminiPrediction(
                    question_id=record.question_id,
                    episode_id=record.episode_id,
                    question_type=record.question_type,
                    gold_answer=record.answer,
                    predicted_answer="ERR",
                    predicted_letter="ERR",
                    correct=False,
                    reasoning=f"ERROR: {e}",
                    raw_response="",
                )
            predictions.append(prediction)
            f.write(json.dumps(prediction_to_dict(prediction), ensure_ascii=False) + "\n")
            f.flush()

    summary = summarize(selected, predictions)
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
