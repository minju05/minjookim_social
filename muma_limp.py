from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import asdict, dataclass

from openai import OpenAI

from muma_data import QuestionRecord


@dataclass(frozen=True)
class LIMPPrediction:
    question_id: str
    episode_id: str
    question_type: str
    gold_answer: str
    predicted_answer: str
    predicted_letter: str
    correct: bool
    reasoning: str


def build_client(api_key: str | None) -> OpenAI:
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set.")
    return OpenAI(api_key=api_key)


def extract_name_from_question(client: OpenAI, question: str) -> list[str]:
    prompt = """You will read a question asking about a person's mental state or actions. From the prompt and options, extract any name of the people you encountered. Determine the person whose mental state or action the question is asking about. Produce your output in this form: [main person's name, name2, name3, ...]. Do not record names appearing multiple times, and do not give any extra information. An example question is like this:
    Example Question: Given that Emma has seen David walking to school yesterday, what will Emma most likely believe
    A David will walk to school tomorrow
    B David will drive to school tomorrow
    C David will not come to school tomorrow
    Example Output: ["Emma", "David"]

    Input Question: {}
    """
    response = client.chat.completions.create(
        messages=[{"role": "system", "content": prompt.format(question)}],
        model="gpt-4o-mini",
        temperature=0.0,
    )
    return ast.literal_eval(response.choices[0].message.content.strip())


def get_choice(client: OpenAI, final_prob: list[float], prompt: str) -> str:
    final_answer = f"""You will read a question with choices and likelihood of each statement for choices in a probability formal. Based on these information, answer the question and only include the letter of choice in your answer. 
    Question: {prompt}
    
    """
    choice_list = ["A", "B", "C", "D", "E"]
    for index, prob in enumerate(final_prob):
        final_answer += f"Probability of statement in choice {choice_list[index]} is True: {prob}\n"
    response = client.chat.completions.create(
        messages=[{"role": "system", "content": final_answer}],
        model="gpt-4o-mini",
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()[0]


def parse_text_info(client: OpenAI, text: str, name: str) -> dict[str, list[str] | None]:
    prompt = """
        You will read a piece of text describing actions of some number of people with distinctive names. You will also have a name, which is the name of the person whom you should pay attention to. Summarize the person's actions and utterance separately in a chronological order. Only include the actions and utterance directly taken by the person in the text, and exclude any previous actions mentioned indirectly. If you cannot find either utterance or actions of the person in the text, leave the corresponding section blank. When reading words like "it", replace it with inferred object or location to make actions clearer. Do not include agent's communication as part of it. Organize your answer in this form:
        Actions:
        ["action one", "action two", "action three", ...]
        ...
        Utterance:
        ["utterance one", "utterance two", "utterance three", ...]
        ... 

        Text: {}

        Name: {}
        
    """
    response = client.chat.completions.create(
        messages=[{"role": "system", "content": prompt.format(text, name)}],
        model="gpt-4o-mini",
        temperature=0.0,
    )
    info = response.choices[0].message.content.strip()
    actions_match = re.search(r"Actions:\s*(\[[^\]]*\])", info)
    utterance_match = re.search(r"Utterance:\s*(\[[^\]]*\])", info)
    actions = ast.literal_eval(actions_match.group(1)) if actions_match else []
    utterances = ast.literal_eval(utterance_match.group(1)) if utterance_match else []
    return {
        "action": actions or None,
        "utterance": utterances or None,
    }


def latent_variable_extraction(client: OpenAI, info: dict[str, dict[str, list[str] | None]], question: str) -> tuple[str, dict[str, str]]:
    latent_variable_prompt = """
    You will read a question about agents' mind and ideas, and the initial state of the environment from which agents' are interacting in. Agents' knowledge & belief are about this initial state, but not necessarily changed state after some actions. For each choice, extract one set of second person's belief (make sure to turn it into some statement about the environment state), second person's social goal toward first peron's actions (help, hinder or some similar words of indepedent), and second person's believed first person's physical goal (some arrangement of objects). Organize the answer in this way: A: Belief: contents; Social goal: contents; Believed Goal: contents. B: Belief: contents; Social goal: contents; Believed Goal: contents. C: Belief: contents; Social goal: contents; Believed Goal: contents. Do not include any other information or extra contents. Make sure your answer follow the format requirement, use ";" to separate variables within each choice and end response with ".". Separate contents of "A", "B" and "C" with "."

    Question: {}
"""
    init_state_prompt = """
    You will read one or two person's actions in a list like form. From the actions taken, extract the initial state of the environment before any people act. 
    Check each grab action or synonyms. Describe it in the form "There is a [object grabbed] [on/inside location of grabbing].
    Only include environment states statements. Do not include any other information or extra contents.

    Actions: {}
"""

    action_str = ""
    for name, person_info in info.items():
        if person_info["action"] is not None:
            action_str += f"{name}'s actions:\n"
            for idx, action in enumerate(person_info["action"]):
                action_str += f"{idx + 1}: {action}\n"

    init_response = client.chat.completions.create(
        messages=[{"role": "system", "content": init_state_prompt.format(action_str)}],
        model="gpt-4o-mini",
        temperature=0.0,
    )
    init_state = init_response.choices[0].message.content.strip()

    names = list(info.keys())
    if len(names) > 1 and info[names[1]]["action"] is not None and info[names[1]]["utterance"] is None:
        prompt = f"""
        Consider the action of {names[1]} before {names[0]} act. Check where {names[1]} has put the object to help you determine {names[1]}'s desired location for the object.
        Actions: {info[names[1]]["action"]} 
        """ + latent_variable_prompt
        response = client.chat.completions.create(
            messages=[{"role": "system", "content": prompt.format(question)}],
            model="gpt-4o-mini",
            temperature=0.0,
        )
    else:
        prompt = latent_variable_prompt + """
        State: {}
        """
        response = client.chat.completions.create(
            messages=[{"role": "system", "content": prompt.format(question, init_state)}],
            model="gpt-4o-mini",
            temperature=0.0,
        )
    latent_variables = response.choices[0].message.content.strip()

    def extract_contents(label: str, input_string: str) -> str | None:
        pattern = rf"{label}: (.*?)(?=[A-Z]:|$)"
        match = re.search(pattern, input_string, re.DOTALL)
        return match.group(1).strip() if match else None

    return init_state, {
        "A": extract_contents("A", latent_variables) or "",
        "B": extract_contents("B", latent_variables) or "",
        "C": extract_contents("C", latent_variables) or "",
    }


def parse_latent_var(latent_var: str) -> dict[str, str]:
    return {
        "Belief": re.search(r"Belief:\s*(.*?)(?=; Social goal)", latent_var).group(1),
        "Social goal": re.search(r"Social goal:\s*(.*?)(?=; Believed Goal)", latent_var).group(1),
        "Believed Goal": re.search(r"Believed Goal:\s*(.*)", latent_var).group(1),
    }


def compute_prob_utterance(
    client: OpenAI,
    name_agent_0: str,
    name_agent_1: str,
    utterance_agent_0: str,
    utterance_agent_1: str,
    social_goal: str,
    belief: str,
    believed_goal: str,
    init_state: str | None,
    exclude: list[str] | None = None,
) -> float:
    exclude = exclude or []
    evaluation_prompt = f"""
    {name_agent_1}'s social goal: {social_goal}
    {name_agent_1}'s belief: {belief}
    """
    if "Believed_Goal" not in exclude:
        evaluation_prompt += f"{name_agent_1}'s belief of {name_agent_0}'s goal: {believed_goal}\n"
    evaluation_prompt += f"{name_agent_0}'s Utterance': {utterance_agent_0}\n"
    if init_state is not None:
        evaluation_prompt += f"Initial state of environment: {init_state}\n"
    evaluation_prompt += f"""
    Based on the information, decide if it is likely for {name_agent_1} to say this word given conditions above. Compare the utterance and the belief of {name_agent_1}. 
    When trying to hinder, {name_agent_1} is likely to give different information with belief. For example, saying that some object is there when {name_agent_1} believe that there is some other things or nothing there, or the object is at a different place.
    Respond with only either A or B:
    {name_agent_1}'s Utterance: {utterance_agent_1}
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


def compute_prob_action(
    client: OpenAI,
    name_agent_0: str,
    name_agent_1: str,
    init_state: str,
    previous_actions: str,
    action: str,
    social_goal: str,
    belief: str,
    believed_goal: str,
) -> float:
    evaluation_prompt = f"""
    Decide if {name_agent_1}'s action is likely with the information provided, respond with only either A or B:
    {name_agent_0}'s social goal: {social_goal}
    {name_agent_1}'s belief: {belief}
    {name_agent_1}'s belief of {name_agent_0}'s goal: {believed_goal}
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


def compute_prob(
    client: OpenAI,
    init_state: str,
    latent_var: str,
    info: dict[str, dict[str, list[str] | None]],
    main_person: str,
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
            probability *= compute_prob_action(
                client,
                other_name,
                main_person,
                init_state,
                previous_actions,
                action,
                social_goal,
                belief,
                believed_goal,
            )
    return probability


def postprocess_visual_actions(client: OpenAI, raw_action_text: str, episode_id: str, text_context: str) -> list[str]:
    first_line = text_context.split("\n")[0]
    inferred_object_context = first_line
    inferred_name = first_line.split(":")[0]

    if int(episode_id) < 4000:
        prompt = f"""
            Read a piece of text, select the object that the person is picking up and moving around, only include the object name in your answer.

            Text: {first_line}
"""
        response = client.chat.completions.create(
            messages=[{"role": "system", "content": prompt}],
            model="gpt-4o-mini",
            temperature=0.0,
        )
        inferred_object_context = response.choices[0].message.content.strip()

        prompt = f"""Read a piece of text, select a person's name from the text. Only output person's name
        Input text: {raw_action_text}
"""
        response = client.chat.completions.create(
            messages=[{"role": "system", "content": prompt}],
            model="gpt-4o-mini",
            temperature=0.0,
        )
        inferred_name = response.choices[0].message.content.strip()

    prompt = """
    Input text: {}
    Additional_information: {}
    Person's name: {}
    You will read some text describe a person's action. The name of the person is given. Only summarize his/her action and ignore actions of other person. Reorganize the person's actions.
    Possible actions include: walk towards somewhere, grab something from somewhere, open some container, close some container, put something somewhere. Only summarize these actions and their synonyms in this form and abandon mismatch actions. Omit peron's name. When mentioning location name, try to infer room the location is inside and include it in the action in form "[container] in [room_name]"
    Check objects mentioned in the Additional Information section. Replace any object mentioned in action with the object appeared in that section
    Formulate your final answer in the following form.
    Actions:
    ["action1", "action2", ....]
    """
    response = client.chat.completions.create(
        messages=[{"role": "system", "content": prompt.format(raw_action_text, inferred_object_context, inferred_name)}],
        model="gpt-4o-mini",
        temperature=0.0,
    )
    actions = response.choices[0].message.content.strip()
    actions_match = re.search(r"Actions:\s*(\[[^\]]*\])", actions)
    if not actions_match:
        raise ValueError(f"Could not parse action list from: {actions}")
    return ast.literal_eval(actions_match.group(1))


def answer_question(
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
    have_utterance = False

    for name in name_list:
        person_info = parse_text_info(client, text, name)
        if person_info["utterance"] is not None:
            person_info["action"] = None
            have_utterance = True
        info[name] = person_info

    if int(record.episode_id) > 4000:
        info[name_list[1]]["action"] = postprocess_visual_actions(client, raw_action_text, record.episode_id, text)
    else:
        if info[name_list[0]]["action"] is None:
            info[name_list[0]]["action"] = postprocess_visual_actions(client, raw_action_text, record.episode_id, text)
        else:
            info[name_list[1]]["action"] = postprocess_visual_actions(client, raw_action_text, record.episode_id, text)

    init_state, latent_var_options = latent_variable_extraction(client, info, question_with_choices)
    prob_list = []
    for choice in ["A", "B", "C"]:
        prob_list.append(compute_prob(client, init_state, latent_var_options[choice], info, name_list[0]))

    max_prob = max(prob_list)
    exp_probs = [math.exp(prob - max_prob) for prob in prob_list]
    total = sum(exp_probs)
    final_prob = [prob / total for prob in exp_probs]
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
        reasoning="; ".join(
            [
                f"A={final_prob[0]:.4f}",
                f"B={final_prob[1]:.4f}",
                f"C={final_prob[2]:.4f}",
            ]
        ),
    )


def prediction_to_dict(prediction: LIMPPrediction) -> dict[str, object]:
    return asdict(prediction)
