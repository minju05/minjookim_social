from __future__ import annotations

import json
import math
from dataclasses import asdict

from openai import OpenAI

from muma_data import QuestionRecord
from muma_limp import (
    LIMPPrediction,
    build_client,
    compute_prob_utterance,
    extract_name_from_question,
    get_choice,
    latent_variable_extraction,
    parse_latent_var,
    parse_text_info,
    postprocess_visual_actions,
    prediction_to_dict,
)


def estimate_perspective_isolated_belief(
    client: OpenAI,
    agent_j: str,
    agent_i: str,
    text: str,
    init_state: str,
) -> str:
    """
    Estimate b_j: what agent_j believes about the world state, based only on
    what agent_j could directly observe (perspective-isolated).

    In the I-POMDP Level-1 formulation, is_i = (s, b_j, g_j).
    LIMP implicitly processes b_j through shared context; Module A estimates
    it explicitly so the scoring can cross-check each hypothesis against b_j.

    This is used as agent_i's "belief of agent_j's belief" in the IMP prompt.
    """
    prompt = f"""You will read a description of a social interaction between {agent_i} and {agent_j}.
Your task is to determine what {agent_j} currently believes about the location or state of the relevant objects.

CRITICAL CONSTRAINT: {agent_j} can only know things they personally observed or did.
Include:
- {agent_j}'s own actions and what they perceived directly
- Statements {agent_i} made in {agent_j}'s presence
- Actions {agent_i} performed while {agent_j} was present and watching

Exclude:
- Any actions {agent_i} took when {agent_j} was absent or could not be watching
- Narrator-level knowledge that {agent_j} would not have access to

Initial environment state: {init_state}
Interaction text: {text}

Based only on {agent_j}'s direct observations, what does {agent_j} currently believe about where the key objects are?
Answer in 1-2 sentences starting with: "{agent_j} believes that ..."
"""
    response = client.chat.completions.create(
        messages=[{"role": "system", "content": prompt}],
        model="gpt-4o-mini",
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()


def compute_prob_action_a(
    client: OpenAI,
    name_agent_0: str,
    name_agent_1: str,
    init_state: str,
    previous_actions: str,
    action: str,
    social_goal: str,
    belief: str,
    believed_goal: str,
    b_hat: str,
) -> float:
    """
    Like LIMP's compute_prob_action, but adds b_hat (Module A: perspective-
    isolated belief estimate) as explicit conditioning alongside the hypothesis
    belief. This lets the LLM cross-check whether the hypothesis is consistent
    with what the agent could actually have observed.
    """
    evaluation_prompt = f"""
    Decide if {name_agent_1}'s action is likely with the information provided, respond with only either A or B:
    {name_agent_0}'s social goal: {social_goal}
    {name_agent_1}'s belief: {belief}
    {name_agent_1}'s belief of {name_agent_0}'s goal: {believed_goal}
    {name_agent_1}'s belief of {name_agent_0}'s current belief about object locations: {b_hat}
    Initial state: {init_state}
    Check {name_agent_0}'s action to get the location of object when {name_agent_1} starts to act.
    When {name_agent_1} tries to hinder, it's likely to grab object from its believed goal location for other agent, and unlikely to move objects to the believed goal location
    When {name_agent_1} tries to help, it's likely to grab object from somewhere else and put it to believed goal location, and unlikely to grab object from believed goal location
    Walking towards or grabbing from some unrelated location should be considered likely
    Previous Actions: {previous_actions}
    {name_agent_1}'s Action: {action}
    A) Likely
    B) Unlikely
"""
    response = client.chat.completions.create(
        messages=[{"role": "system", "content": evaluation_prompt}],
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
    return math.exp(logprob_a) if logprob_a is not None else 0.0


def compute_prob_a(
    client: OpenAI,
    init_state: str,
    latent_var: str,
    info: dict[str, dict[str, list[str] | None]],
    main_person: str,
    b_hat: str,
) -> float:
    latent_vars = parse_latent_var(latent_var)
    belief = latent_vars["Belief"]
    social_goal = latent_vars["Social goal"]
    believed_goal = latent_vars["Believed Goal"]
    names = list(info.keys())
    other_name = [name for name in names if name != main_person][0]
    probability = 1.0

    if info[main_person]["utterance"] is not None:
        probability = compute_prob_utterance(
            client,
            other_name,
            main_person,
            info[other_name]["utterance"][0],
            info[main_person]["utterance"][0],
            social_goal,
            belief,
            believed_goal,
            None,
            exclude=["Believed_Goal"],
        )

    if info[main_person]["action"] is not None:
        for index, action in enumerate(info[main_person]["action"]):
            previous_actions = f"{other_name}'s actions:\n"
            for other_action in info[other_name]["action"] or []:
                previous_actions += other_action + "\n"
            previous_actions += f"{main_person}'s actions:\n"
            for prior_action in (info[main_person]["action"] or [])[:index]:
                previous_actions += prior_action + "\n"
            probability *= compute_prob_action_a(
                client,
                other_name,
                main_person,
                init_state,
                previous_actions,
                action,
                social_goal,
                belief,
                believed_goal,
                b_hat,
            )
    return probability


def answer_question_a(
    client: OpenAI,
    record: QuestionRecord,
    raw_action_text: str,
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

    main_person = name_list[0]
    other_name = name_list[1]
    init_state, latent_var_options = latent_variable_extraction(client, info, question_with_choices)

    # Module A: estimate b_j (other agent's perspective-isolated belief),
    # used as main_person's "belief of other agent's belief" in IMP scoring.
    b_hat = estimate_perspective_isolated_belief(
        client, other_name, main_person, text, init_state
    )

    prob_list = []
    for choice in ["A", "B", "C"]:
        prob_list.append(
            compute_prob_a(client, init_state, latent_var_options[choice], info, main_person, b_hat)
        )

    max_prob = max(prob_list)
    exp_probs = [math.exp(p - max_prob) for p in prob_list]
    total = sum(exp_probs)
    final_prob = [p / total for p in exp_probs]
    model_choice = get_choice(client, final_prob, question_with_choices)
    index = ord(model_choice) - ord("A")
    predicted_answer = record.choices[index]

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
            f"b_hat={b_hat[:100]}",
        ]),
    )