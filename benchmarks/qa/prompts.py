"""
Prompts for the end-to-end QA eval (benchmarks/qa/eval_plan/prereg.md).

* ANSWER_PROMPT: one neutral prompt for every arm (ours). It carries no evaluation
  words, so the answering model sees an ordinary memory-assistant request.
* LME_JUDGE: LongMemEval's official answer-check templates, verbatim from
  xiaowu0162/LongMemEval src/evaluation/evaluate_qa.py (MIT licence). Also used
  as LoCoMo's strict judge (J-strict).
* MEM0_JUDGE: Mem0's 2026 unified LoCoMo judge, verbatim from
  mem0ai/memory-benchmarks benchmarks/locomo/prompts.py (Apache-2.0). Lenient by
  design (one matching list item, dates within 14 days, durations within 50%);
  used only as J-lenient, to sit beside published LoCoMo numbers.
"""

ANSWER_PROMPT = """You are a helpful assistant with long-term memory of past conversations. Below are memories from those conversations, oldest first, each with the date it happened.

{memories}

{date_line}Answer the question using these memories. Work out relative dates ("yesterday", "last week") from the date of the memory they appear in. Answer in a short phrase or sentence. If the memories do not contain the answer, say that you don't know.

Question: {question}
Answer:"""

# prereg v7: answer-prompt variants. Only the instruction paragraph differs; p0 is ANSWER_PROMPT
# exactly, and a queued row picks its prompt with a "prompt" field (absent = p0).
_ANSWER_HEAD, _ANSWER_TAIL = ANSWER_PROMPT.split("{date_line}")[0] + "{date_line}", "\n\nQuestion: {question}\nAnswer:"
_DATES = 'Work out relative dates ("yesterday", "last week") from the date of the memory they appear in.'
ANSWER_INSTRUCTIONS = {
    "p0": f"Answer the question using these memories. {_DATES} Answer in a short phrase or sentence. "
          "If the memories do not contain the answer, say that you don't know.",
    "p1": f"Answer the question using these memories. {_DATES} If the question asks about facts or events, "
          "answer in a short phrase or sentence, and if the memories do not contain the answer, say that you "
          "don't know. If it asks for advice, suggestions or an opinion, give a helpful answer that draws on "
          "what the memories say about the user: their preferences, past experiences, plans and circumstances.",
    "p2": f"Answer the question based on the relevant memories. {_DATES}",  # as LongMemEval's own reader
    # prereg v8: P1 with LongMemEval's step-by-step reader instruction (run_generation.py, CoT option)
    "p1s": f"Answer the question using these memories. {_DATES} Answer step by step: first extract all the "
           "relevant information from the memories, and then reason over it to get the answer. End with your "
           "answer: if the question asks about facts or events, a short phrase or sentence, or that you don't "
           "know if the memories do not contain it; if it asks for advice, suggestions or an opinion, a helpful "
           "answer that draws on what the memories say about the user: their preferences, past experiences, "
           "plans and circumstances.",
}
ANSWER_PROMPTS = {k: _ANSWER_HEAD + v + _ANSWER_TAIL for k, v in ANSWER_INSTRUCTIONS.items()}
assert ANSWER_PROMPTS["p0"] == ANSWER_PROMPT

LME_JUDGE = {
    "generic": "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.",
    "temporal-reasoning": "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.",
    "knowledge-update": "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.",
    "single-session-preference": "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.",
    "abstention": "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."
}

MEM0_JUDGE_SYSTEM = "You are evaluating conversational AI memory recall. Return JSON only with the format requested."

MEM0_JUDGE = "Label the generated answer as CORRECT or WRONG.\n\n## Rules\n\n1. **PARTIAL CREDIT**: If the generated answer includes AT LEAST ONE correct item from the gold answer's list, mark CORRECT. Getting 1 out of 2, 2 out of 4, etc. is always acceptable. Only mark WRONG if NONE of the gold answer items appear.\n\n2. **PARAPHRASES COUNT**: Same concept in different words is CORRECT. \"Chocolate raspberry tart\" = \"chocolate cake with raspberries\". \"Shelter meal service\" = \"volunteering at a homeless shelter\". Emotions and sentiments in the same positive/negative family count as paraphrases: \"proud\" = \"fulfilled\" = \"accomplished\"; \"huge success\" = \"relieved\" = \"thrilled\" (all express positive achievement). Judge semantic meaning, not exact wording.\n\n3. **EXTRA DETAIL IS FINE**: A longer answer that includes the gold answer's key facts plus additional information is CORRECT. Never penalize for being more detailed or specific. If the generated answer adds extra descriptive details beyond the gold answer while still referencing the same core entity or concept, mark CORRECT.\n\n4. **DATE TOLERANCE**: Dates within 14 days of each other are CORRECT. Durations within 50% are CORRECT (e.g., \"5 months\" matches \"six months\"; \"19 days\" matches \"two weeks\"). Relative dates (\"few days before November\") match specific dates in the same window. A specific date (e.g., \"February 2020\") that is consistent with a vague reference (e.g., \"a few years ago\" relative to 2023) is CORRECT. Converting \"last year\" to the actual year (e.g., \"2022\" when conversations are in 2023) is CORRECT.\n\n5. **SEMANTIC OVERLAP**: Judge whether the generated answer addresses the same topic and captures the core idea of the gold answer. Different wording, phrasing, or level of detail should not result in WRONG if the underlying concept matches. For EMOTIONS and FEELINGS questions, answers expressing sentiments in the same valence (positive/negative) about the same event are CORRECT \u2014 do not require the exact same emotion word.\n\n6. **SAME REFERENT**: If the generated answer mentions or references the same named entity, character, person, or concept as the gold answer, mark CORRECT \u2014 even if the generated answer provides a different physical description or includes additional details. The key question is: does the generated answer identify the same core entity? If yes, it is CORRECT.\n\n7. **FOCUS ON KNOWLEDGE, NOT WORDING**: The goal is to assess whether the system recalled the right fact. Minor differences in specificity, phrasing, or scope should not result in WRONG. Only mark WRONG when the generated answer demonstrates a genuinely different or incorrect understanding.\n\n## ONLY mark WRONG if:\n- The generated answer contains ZERO correct items from the gold answer\n- The answer addresses a completely different topic\n\n## Question\nQuestion: {question}\nGold answer: {answer}\nGenerated answer: {response}\n\nReturn JSON with \"reasoning\" (one sentence) and \"label\" (CORRECT or WRONG). Do NOT include both labels."



def lme_judge_prompt(qtype: str, question: str, answer: str, response: str, abstention: bool = False) -> str:
    """LongMemEval's get_anscheck_prompt, same template choice per question type."""
    if abstention:
        key = "abstention"
    elif qtype in ("temporal-reasoning", "knowledge-update", "single-session-preference"):
        key = qtype
    else:
        key = "generic"
    return LME_JUDGE[key].format(question, answer, response)


def mem0_judge_prompt(question: str, answer: str, response: str) -> str:
    return MEM0_JUDGE.format(question=question, answer=answer, response=response)
