# Description: Contains the prompt configuration and helper functions for the policy classification task
# These are currently setup to permute combinations of personas and encouragements, and then customizes
# them for a specific issue area. 

personas = ['You are an expert social scientist with a PhD in political science. ',
            'You are highly intelligent. ',
            'You are a professor of economics. ']

encouragements = ['This will be fun! ',
                  'Think carefully about your answer. ',
                  'I really need your help. ']

general = '''You are conducting research on the policy positions that
European parties in parliamentary democracies take in their political
manifestos. Political manifestos are documents parties produce to explain
their policy positions to voters in an election. For the following text of
a party manifesto, please classify the party position on the overall
orientation of the party '''

ending = '''Only give the score with no explanation. Return it in the form of 
a number without any Markdown formatting like: 5 or 1. If you are uncertain
on the score give a score of NA.
'''

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

policy_areas = {'european_union': 'the European Union and European integration',
                'taxation': 'taxation, public spending and government services',
                'lifestyle': 'social and lifestyle policies including issues like homosexuality and DEI issues',
                'immigration': 'immigration and border control',
                'environment': 'environmental protection and the trade-offs with economic growth',
                'decentralization': 'political decentralization and the role of regional governments'}

def get_prompts(issue_area, manifesto):
    """
    Generate prompts for analyzing a manifesto based on the given issue area.

    Args:
        issue_area (str): The issue area for which prompts are generated.
        manifesto (str): The manifesto text to be analyzed.

    Returns:
        list: A list of prompts, where each prompt is a list of dictionaries with 'role' and 'content' keys.
    """
    
    system_prompts = [ (persona + encouragement + general + policy_scales[issue_area] + ending).replace('\n', ' ') for persona in personas for encouragement in encouragements]
    prompts = []
    for system_prompt in system_prompts:
        prompts.append([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Analyze the following text:\n\n{manifesto}"}
        ]) 

    return prompts