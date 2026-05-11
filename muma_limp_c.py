from __future__ import annotations

import json
import math
from dataclasses import asdict

from openai import OpenAI

from muma_data import QuestionRecord
from muma_limp import (
    LIMPPrediction,
    build_client,
    compute_prob,
    extract_name_from_question,
    get_choice,
    latent_variable_extraction,
    parse_latent_var,
    parse_text_info,
    postprocess_visual_actions,
    prediction_to_dict,
)


def compute_consistency_score(
    client: OpenAI,
    name_agent: str,
    utterance: str,
    subsequent_actions: list[str],
) -> float:
    """Returns P(consistent): probability that utterance matches agent's actual actions."""
    actions_str = "; ".join(subsequent_actions)
    prompt = (
        f"Agent {name_agent}'s utterance: {utterance}\n"
        f"Agent {name_agent}'s subsequent actions: {actions_str}\n\n"
        "Does the utterance truthfully reflect the agent's intentions given their actual actions?\n"
        "A) Consistent\n"
        "B) Inconsistent\n"
        "Respond with only A or B."
    )
    response = client.chat.completions.create(
        messages=[{"role": "system", "content": prompt}],
        model="gpt-4o-mini",
        logprobs=True,
        top_logprobs=5,
        temperature=0.0,
    )
    response_dict = json.loads(response.model_dump_json(indent=2))
    logprob_a = None
    for top_logprob in response_dict["choices"][0]["logprobs"]["content"][0]["top_logprobs"]:
        if top_logprob["token"] == "A":
            logprob_a = top_logprob["logprob"]
    return math.exp(logprob_a) if logprob_a is not None else 0.5


def _social_goal_type(social_goal: str) -> str:
    lower = social_goal.lower()
    if any(w in lower for w in ("hinder", "prevent", "obstruct")):
        return "hinder"
    if any(w in lower for w in ("help", "assist", "cooperat", "support")):
        return "help"
    return "independent"


def compute_prob_with_consistency(
    client: OpenAI,
    init_state: str,
    latent_var: str,
    info: dict[str, dict[str, list[str] | None]],
    main_person: str,
    consistency_score: float | None,
    lambda_weight: float,
) -> float:
    base = compute_prob(client, init_state, latent_var, info, main_person)

    if consistency_score is None:
        return base

    latent_vars = parse_latent_var(latent_var)
    goal_type = _social_goal_type(latent_vars["Social goal"])

    if goal_type == "help":
        s_c = consistency_score
    elif goal_type == "hinder":
        s_c = 1.0 - consistency_score
    else:
        return base

    s_c = max(s_c, 1e-9)
    return base * (s_c ** lambda_weight)


def answer_question_c(
    client: OpenAI,
    record: QuestionRecord,
    raw_action_text: str,
    lambda_weight: float = 1.0,
) -> LIMPPrediction:
    question_with_choices = "\n".join(
        [
            record.question,
            f"A) {record.choices[0]}",
            f"B) {record.choices[1]}",
            f"C) {record.choices[2]}",
        ]
    )
    name_list = extract_name_from_question(client, question_with_choices)
    text = record.text_context or ""
    info: dict[str, dict[str, list[str] | None]] = {}

    for name in name_list:
        person_info = parse_text_info(client, text, name)
        if person_info["utterance"] is not None:
            person_info["action"] = None
        info[name] = person_info

    if int(record.episode_id) > 4000:
        info[name_list[1]]["action"] = postprocess_visual_actions(client, raw_action_text, record.episode_id, text)
    else:
        if info[name_list[0]]["action"] is None:
            info[name_list[0]]["action"] = postprocess_visual_actions(client, raw_action_text, record.episode_id, text)
        else:
            info[name_list[1]]["action"] = postprocess_visual_actions(client, raw_action_text, record.episode_id, text)

    # Module C: consistency score when main agent has both utterance (from text) and visual action
    main_person = name_list[0]
    consistency_score: float | None = None
    main_utterance = info[main_person].get("utterance")
    main_action = info[main_person].get("action")
    if main_utterance and main_action:
        consistency_score = compute_consistency_score(
            client,
            main_person,
            main_utterance[0],
            main_action,
        )

    init_state, latent_var_options = latent_variable_extraction(client, info, question_with_choices)

    prob_list = []
    for choice in ["A", "B", "C"]:
        prob_list.append(
            compute_prob_with_consistency(
                client,
                init_state,
                latent_var_options[choice],
                info,
                main_person,
                consistency_score,
                lambda_weight,
            )
        )

    max_prob = max(prob_list)
    exp_probs = [math.exp(p - max_prob) for p in prob_list]
    total = sum(exp_probs)
    final_prob = [p / total for p in exp_probs]
    model_choice = get_choice(client, final_prob, question_with_choices)
    index = ord(model_choice) - ord("A")
    predicted_answer = record.choices[index]

    consistency_str = f"{consistency_score:.4f}" if consistency_score is not None else "N/A"
    return LIMPPrediction(
        question_id=record.question_id,
        episode_id=record.episode_id,
        question_type=record.question_type,
        gold_answer=record.answer,
        predicted_answer=predicted_answer,
        predicted_letter=model_choice,
        correct=predicted_answer == record.answer,
        reasoning="; ".join([
            f"A={final_prob[0]:.4f}",
            f"B={final_prob[1]:.4f}",
            f"C={final_prob[2]:.4f}",
            f"consistency={consistency_str}",
        ]),
    )