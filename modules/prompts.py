import yaml
import os

import pandas as pd

from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage, AIMessage


def load_prompts(what, version=None, folder=None):
    """
    Load prompts from external yaml file.
    The filename should be in the following format: prompts-<summarize|analyze>[__{version}].yaml
        e.g. prompts-summarize__test.yaml, prompts_summarize.yaml

    Args:
        what (str): One of [summarize, analyze].
        version (str): The version of the prompts to load, excluding the leading double underscores.
        folder (str): The folder where to find the prompts file.
    """
    if folder is None:
        # Use default folder
        folder = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), "prompts")

    prompts_filepath = os.path.join(folder, f"prompts-{what}{'' if not version else f'__{version}'}.yaml")
    with open(prompts_filepath, "r", encoding="utf-8") as f:
        prompts = yaml.safe_load(f.read())

    return prompts


def get_prompts(issue_area, text, override_persona_and_encouragement=None):
    """
    Generate prompts for analyzing a manifesto based on the given issue area.
    If override_persona_and_encouragement are provided, use it in the prompt.
    Otherwise, permutes the personas and encouragements to create a variety of prompts
    for the same issue area and text.

    Args:
        issue_area (str): The issue area for which prompts are generated.
        text (str): The text to be analyzed.
        override_persona_and_encouragement (tuple): The persona and encouragement to use.

    Returns:
        list: A list of prompts, where each prompt is a list of dictionaries with 'role' and 'content' keys.
    """

    prompts_analyze = load_prompts("analyze")
    personas = prompts_analyze["personas"]
    encouragements = prompts_analyze["encouragements"]
    policy_scales = prompts_analyze["policy_scales"]
    system_template_string = prompts_analyze["system_template_string"]
    human_template_string = prompts_analyze["human_template_string"]

    system_template = PromptTemplate(template=system_template_string)
    human_template = PromptTemplate(template=human_template_string)

    if override_persona_and_encouragement is not None:
        idx_persona, idx_encouragement = override_persona_and_encouragement
        return [[
            SystemMessage(content=system_template.format(persona=personas[idx_persona],
                                                         encouragement=encouragements[idx_encouragement],
                                                         policy_scale=policy_scales[issue_area])),
            HumanMessage(content=human_template.format(text=text))
        ]]
    else:
        prompts = []
        for persona in personas:
            for encouragement in encouragements:
                prompts.append([
                    SystemMessage(content=system_template.format(persona=persona, encouragement=encouragement,
                                                                 policy_scale=policy_scales[issue_area])),
                    HumanMessage(content=human_template.format(text=text))
                ])
        return prompts


def get_few_shot_prompt(
        issue_area, text,
        example_csv_filepath=None,
        # example_search_path=None
):
    """
    Generate prompt for analyzing a manifesto based on the given issue area.

    This function is used to generate prompts for the few-shot learning task pulling
    examples from the CHES dataset. It looks for a file created in the notebook:
    llm_political_analysis/notebooks/few_shot_prompt_setup.ipynb

    Args:
        issue_area (str): The issue area for which prompts are generated.
        text (str): The text to be analyzed.
        example_csv_filepath (str): Path of example metadata csv file.
        example_search_path (str): Path of example summaries search directory.

    Returns:
        list: A list of messages for the few-shot learning task.
    """
    if not example_csv_filepath:
        example_csv_filepath = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), "data", "few_shot_examples.csv")
    # if not example_csv_filepath:
    #     example_csv_filepath = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), "data", "few_shot_prompts", "few_shot_prompt_setup.csv")
    # if not example_search_path:
    #     example_search_path = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), "data", "few_shot_prompts")

    prompts_analyze = load_prompts("analyze")
    personas = prompts_analyze["personas"]
    encouragements = prompts_analyze["encouragements"]
    policy_scales = prompts_analyze["policy_scales"]
    system_template_string = prompts_analyze["system_template_string"]
    human_template_string = prompts_analyze["human_template_string"]

    # examples = pd.read_csv('../data/ches_scores/final_prompt_setup_new.csv',
    #                        dtype={'Expert mean': str}, keep_default_na=False)
    examples = pd.read_csv(example_csv_filepath,
                           dtype={'Expert mean': str}, keep_default_na=False)
    issue_examples = examples[examples['issue'] == issue_area]

    system_template = PromptTemplate(template=system_template_string)
    human_template = PromptTemplate(template=human_template_string)
    ai_template = PromptTemplate(template='{text}')

    # It was decided after prompt varation analysis that the first persona and second encouragement
    # were sufficient.

    persona = personas[0]
    encouragement = encouragements[1]

    prompt = [SystemMessage(content=system_template.format(
        persona=persona, encouragement=encouragement, policy_scale=policy_scales[issue_area]))]
    for example in issue_examples.iterrows():
        # summary_file_name = example[1]['Calibration File']
        summary  = example[1]['summary']
        # with open(os.path.join(example_search_path, summary_file_name), 'r') as summary_file:
        #     summary = summary_file.read()
        prompt.append(HumanMessage(
            content=human_template.format(text=summary)))
        prompt.append(AIMessage(content=ai_template.format(
            text=example[1]['Expert mean'])))
    prompt.append(HumanMessage(
        content=human_template.format(text=text)))

    return prompt
