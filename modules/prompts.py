import json
import os

from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage

with open(
        os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), "prompts", "prompts-analyze.json"),
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
