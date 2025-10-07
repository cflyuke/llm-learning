import re


def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = text.split("</answer>")[0]
    return answer.strip()

def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    gen_answers = [extract_xml_answer(response) for response in responses]
    return [2.0 if r == a else 0.0 for r, a in zip(gen_answers, answer)]


def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    gen_answers = [extract_xml_answer(response) for response in responses]
    return [0.5 if r.isdigit() else 0.0 for r in gen_answers]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>\n?$"
    reponses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, re.DOTALL) for r in reponses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"<think>.*?</think>.*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, r, re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<think>\n") == 1:
        count += 0.125
    if text.count("\n</think>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001  # 不以</answer>结尾扣除部分奖励分数
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001  # 不以</answer>结尾扣除部分奖励分数
    return count

def xmlcount_reward_func(completions, **kwargs):
    responses = [completion[0]["content"] for completion in completions]
    return [count_xml(r) for r in responses]

REWARD_FUNCS = {
    'correctness_reward_func': correctness_reward_func,
    'int_reward_func': int_reward_func,
    'strict_format_reward_func': strict_format_reward_func,
    'soft_format_reward_func': soft_format_reward_func,
    'xmlcount_reward_func': xmlcount_reward_func
}