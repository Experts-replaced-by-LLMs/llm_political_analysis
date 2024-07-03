import os

import pandas as pd
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage, AIMessage

personas = ['You are an expert social scientist with a PhD in political science. ',
            'You are highly intelligent. ',
            'You are a professor of economics. ']

encouragements = ['This will be fun! ',
                  'Think carefully about your answer. ',
                  'I really need your help. ']

policy_scales = {'european_union': '''toward the European Union. Classify the manifesto
on this policy using a seven point scale, where a 1 means strongly opposed,
a 2 means opposed, a 3 means somewhat opposed, a 4 means neutral, a 5 means
somewhat in favor, a 6 means in favor, and a 7 means strongly in favor of
the European Union. If the text of the manifesto does not provide a clear
position on the European Union, return the result of NA (meaning non-
applicable).''',

'taxation': '''toward Spending versus Taxation. Classify the
manifesto on this policy using a ten point scale, where a 1 means strongly
favors improving public services, a 5 means the party takes a position in
the middle or balanced position between raising spending and reducing
taxes, and a 10 means strongly favors reducing taxes. If the text of the
manifesto does not provide a clear position on spending vs taxation, return
the result of NA (meaning non-applicable).''',

'lifestyle': '''toward Social and Lifestyle policies (for example,
homosexuality). Classify the manifesto on this policy using a ten point
scale, where a 1 means strongly supports liberal social policies, a 5 means
the party takes a position in the middle or balanced position on social
policies, and a 10 means strongly opposes liberal social policies. If the
text of the manifesto does not provide a clear position on social and
lifestyle policies, return the result of NA (meaning non-applicable).''',

'immigration': '''toward Immigration. Classify the manifesto on
this policy using a ten point scale, where a 1 means strongly opposes a
tough immigration policy and wants more open borders, a 5 means the party
takes a position in the middle or balanced position on immigration, and a
10 means strongly favors a tough immigration policy that reduces the number
of immigrants to the country. If the text of the manifesto does not
provide a clear position on immigration, return the result of NA (meaning
non-applicable).''',

'environment': '''toward the Environment. Classify the manifesto on
this policy using a ten point scale, where a 1 means strongly supports
environmental protection even at the cost of economic growth, a 5 means the
party takes a position in the middle or balanced position between
protecting the environment and encouraging economic growth, and a 10 means
strongly supports economic growth even at the cost of environmental
protection. If the text of the manifesto does not provide a clear position
on the environment, return the result of NA (meaning non-applicable).''',

'decentralization': '''toward Political Decentralization to Regions.
Classify the manifesto on this policy using a ten point scale, where a 1
means strongly strongly favors political decentralization, a 5 means the
party takes a position in the middle or balanced position on
decentralization, and a 10 means strongly opposes political
decentralization. If the text of the manifesto does not provide a clear
position on political decentralization, return the result of NA (meaning
non-applicable).''',
}

# this brief description of the policy areas is used in the summarization module
# I colocated it here to be proximate to the scales. 
policy_areas = {'european_union': 'the European Union and European integration',
                'taxation': 'taxation, public spending and the trade offs between them',
                'lifestyle': 'social and lifestyle policies including issues like homosexuality and DEI issues',
                'immigration': 'immigration and border control',
                'environment': 'environmental protection and the trade-offs with economic growth',
                'decentralization': 'political decentralization and the role of regional governments'}

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
    
    system_template_string = '''
    {persona} {encouragement}
    You are conducting research on the policy positions that
    European parties in parliamentary democracies take in their political
    manifestos. Political manifestos are documents parties produce to explain
    their policy positions to voters in an election. For the following text of
    a party manifesto, please classify the party position on the overall
    orientation of the party {policy_scale}
    Only give the score with no explanation. Return it in the form of 
    a number without any Markdown formatting, it should look like: 5 or 1. 
    If you are uncertain on the score give a score of NA.
    '''
    
    system_template = PromptTemplate(template=system_template_string)
    human_template = PromptTemplate(template='Analyze the following political text:\n{text}')

    prompts = []
    for persona in personas:
        for encouragement in encouragements:
            prompts.append([
            SystemMessage(content=system_template.format(persona=persona, encouragement=encouragement, policy_scale=policy_scales[issue_area])),
            HumanMessage(content=human_template.format(text=text))
            ]) 

    return prompts


def create_cleaned_examples():
    '''
    A simple function to load the cleaned examples from the data folder.
    '''

    summary_list = os.listdir('../data/summaries/')
    calibration_summary_list = [x for x in summary_list if 'Calibration' in x]

    ches_results = pd.read_excel('../data/ches_scores/LLM scores vs CHES.xlsx')[['Document','Status','Dimension','Expert mean']]
    ches_results = ches_results[~ches_results['Status'].isna()]
    ches_results['Calibration File'] = ches_results['Document'].apply(lambda x: next((f for f in calibration_summary_list if x in f), None))

    issue_map_reversed = {'EU': 'european_union',
                 'TaxSpend': 'taxation',
                 'SocialLifestyle': 'lifestyle',
                 'Immigration': 'immigration',
                 'Environment': 'environment',
                 'Regions': 'decentralization'}
    
    ches_results['issue'] = ches_results['Dimension'].apply(lambda x: issue_map_reversed[x])

    final_prompt_setup = ches_results[['issue','Calibration File','Expert mean']].where(ches_results['Calibration File'].notna()).dropna()
    final_prompt_setup['Expert mean'] = pd.to_numeric(final_prompt_setup['Expert mean'], errors='coerce').round(0).apply(lambda x: str(int(x)) if not pd.isna(x) else 'NA')
    final_prompt_setup.to_csv('../data/ches_scores/final_prompt_setup.csv', index=False)


def get_few_shot_prompts(issue_area, text):
    """
    Generate prompts for analyzing a manifesto based on the given issue area.
    It permutes the personas and encouragements to create a variety of prompts
    for the same issue area and text.

    This function is used to generate prompts for the few-shot learning task pulling
    examples from the CHES dataset. It looks for a file created by create_cleaned_examples. 

    Args:
        issue_area (str): The issue area for which prompts are generated.
        text (str): The text to be analyzed.

    Returns:
        list: A list of prompts, where each prompt is a list of dictionaries with 'role' and 'content' keys.
    """
    examples = pd.read_csv('../data/ches_scores/final_prompt_setup.csv', dtype={'Expert mean': str}, keep_default_na=False )
    issue_examples = examples[examples['issue'] == issue_area]

    system_template_string = '''
    {persona} {encouragement}
    You are conducting research on the policy positions that
    European parties in parliamentary democracies take in their political
    manifestos. Political manifestos are documents parties produce to explain
    their policy positions to voters in an election. For the following text of
    a party manifesto, please classify the party position on the overall
    orientation of the party {policy_scale}
    Only give the score with no explanation. Return it in the form of 
    a number without any Markdown formatting, it should look like: 5 or 1. 
    If you are uncertain on the score give a score of NA. Here are some examples:
    '''
    
    system_template = PromptTemplate(template=system_template_string)
    human_template = PromptTemplate(template='Analyze the following political text:\n{text}')
    ai_template = PromptTemplate(template='{text}')


    prompts = []
    for persona in personas:
        for encouragement in encouragements:
            prompt = [SystemMessage(content=system_template.format(persona=persona, encouragement=encouragement, policy_scale=policy_scales[issue_area]))]
            for example in issue_examples.iterrows():
                summary_file_name = example[1]['Calibration File']
                with open(f'../data/summaries/{summary_file_name}', 'r') as summary_file:
                    summary = summary_file.read()
                prompt.append(HumanMessage(content=human_template.format(text=summary)))
                prompt.append(AIMessage(content=ai_template.format(text=example[1]['Expert mean'])))
            prompt.append(HumanMessage(content=human_template.format(text=text)))
            prompts.append(prompt)


    return prompts