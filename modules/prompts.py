import os
import json

import pandas as pd
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage, AIMessage

with open(
        os.path.join(os.path.abspath(os.path.dirname(
            os.path.dirname(__file__))), "prompts", "prompts-analyze.json"),
        "r", encoding="utf-8"
) as f:
    prompts_analyze = json.loads(f.read())

personas = prompts_analyze["personas"]
encouragements = prompts_analyze["encouragements"]
policy_scales = prompts_analyze["policy_scales"]

# this brief description of the policy areas is used in the summarization module
# I colocated it here to be proximate to the scales.
policy_areas = prompts_analyze["policy_areas"]

system_template_string = prompts_analyze["system_template_string"]
human_template_string = prompts_analyze["human_template_string"]


def get_prompts(issue_area, text):
    """
    Generate prompts for analyzing a manifesto based on the given issue area.
    It permutes the personas and encouragements to create a variety of prompts
    for the same issue area and text.

    Args:
        issue_area (str): The issue area for which prompts are generated.
        text (str): The text to be analyzed.

    Returns:
        list: A list of prompts, where each prompt is a list of dictionaries with 'role' and 'content' keys.
    """

    system_template = PromptTemplate(template=system_template_string)
    human_template = PromptTemplate(template=human_template_string)

    prompts = []
    for persona in personas:
        for encouragement in encouragements:
            prompts.append([
                SystemMessage(content=system_template.format(
                    persona=persona, encouragement=encouragement, policy_scale=policy_scales[issue_area])),
                HumanMessage(content=human_template.format(text=text))
            ])

    return prompts


def create_cleaned_examples():
    '''
    A simple function to load the cleaned examples from the data folder.
    '''

    summary_list = os.listdir('../data/summaries/')
    calibration_summary_list = [x for x in summary_list if 'Calibration' in x]

    ches_results = pd.read_excel('../data/ches_scores/LLM scores vs CHES.xlsx')[
        ['Document', 'Status', 'Dimension', 'Expert mean']]
    ches_results = ches_results[~ches_results['Status'].isna()]
    ches_results['Calibration File'] = ches_results['Document'].apply(
        lambda x: next((f for f in calibration_summary_list if x in f), None))

    issue_map_reversed = {'EU': 'european_union',
                          'TaxSpend': 'taxation',
                          'SocialLifestyle': 'lifestyle',
                          'Immigration': 'immigration',
                          'Environment': 'environment',
                          'Regions': 'decentralization'}

    ches_results['issue'] = ches_results['Dimension'].apply(
        lambda x: issue_map_reversed[x])

    final_prompt_setup = ches_results[['issue', 'Calibration File', 'Expert mean']].where(
        ches_results['Calibration File'].notna()).dropna()
    final_prompt_setup['Expert mean'] = pd.to_numeric(final_prompt_setup['Expert mean'], errors='coerce').round(
        0).apply(lambda x: str(int(x)) if not pd.isna(x) else 'NA')
    final_prompt_setup.to_csv(
        '../data/ches_scores/final_prompt_setup.csv', index=False)


def get_few_shot_prompt(issue_area, text):
    """
    Generate prompt for analyzing a manifesto based on the given issue area.

    This function is used to generate prompts for the few-shot learning task pulling
    examples from the CHES dataset. It looks for a file created by create_cleaned_examples. 

    Args:
        issue_area (str): The issue area for which prompts are generated.
        text (str): The text to be analyzed.

    Returns:
        list: A list of messages for the few-shot learning task.
    """
    examples = pd.read_csv('../data/ches_scores/final_prompt_setup.csv',
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
        summary_file_name = example[1]['Calibration File']
        with open(f'../data/summaries/{summary_file_name}', 'r') as summary_file:
            summary = summary_file.read()
        prompt.append(HumanMessage(
            content=human_template.format(text=summary)))
        prompt.append(AIMessage(content=ai_template.format(
            text=example[1]['Expert mean'])))
    prompt.append(HumanMessage(
        content=human_template.format(text=text)))

    return prompt
