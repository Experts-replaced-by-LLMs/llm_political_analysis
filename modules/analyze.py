import warnings
from datetime import datetime
import os

import numpy as np
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

from llm_political_analysis.modules import openai_model_list, claude_model_list, gemini_model_list
from llm_political_analysis.modules.prompts import get_prompts, get_few_shot_prompt
from llm_political_analysis.modules.summarize import summarize_file


def validate_score(score):
    return score in ['NA', '1', '2', '3', '4', '5', '6', '7']


def analyze_text(prompt, model, parse_retries=3, max_retries=7, probabilities=False, dry_run=False):
    """
    Analyzes the given text, given a prompt using the specified model.

    Parameters:
    - prompt (list): A list of message objects representing the conversation.
    - model (str): The name or ID of the model to use for analysis.
    - parse_retries (int): The number of times to retry parsing the response. Defaults to 3.
    - max_retries (int): The number of times to retry invoking the model. Defaults to 7, which should be enough
            to handle most TPM rate limits with langchains built in exponential backoff.
    - probabilities (bool): Whether to include token probabilities in the response. Defaults to False. Only works with OpenAI models.
    - dry_run (bool, optional): Don't invoke the LLM api call. Return a mock response for debug and testing. Defaults to False.
    Returns:
    - dict: A dictionary containing the score, any errors and the prompt used. 

    Checked models:
    - OpenAI models: ['gpt-3.5-turbo', 'gpt-4o', 'gpt-4']
    - Claude models: ['claude-3-5-sonnet-20240620', 'claude-3-opus-20240229', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307']
    - Gemini models: ['gemini-1.5-pro-001']

    """
    # Tested models, this list should be updated as new models are added
    # Each service requires an API key to work, stored as an envorinment variable
    # openai_model_list = ['gpt-3.5-turbo', 'gpt-4o', 'gpt-4']
    # claude_model_list = ['claude-3-5-sonnet-20240620', 'claude-3-opus-20240229',
    #                      'claude-3-sonnet-20240229', 'claude-3-haiku-20240307']
    # gemini_model_list = ['gemini-1.5-pro-001']

    if model in openai_model_list:
        llm = ChatOpenAI(temperature=0, max_tokens=150,
                         model_name=model, max_retries=max_retries)
    elif model in claude_model_list:
        llm = ChatAnthropic(temperature=0, max_tokens=150,
                            model_name=model, max_retries=max_retries)
    elif model in gemini_model_list:
        llm = ChatGoogleGenerativeAI(
            temperature=0, max_tokens=150, model=model, max_retries=max_retries)
    else:
        print("You've selected a model that is not available.")
        print(
            f"Please select from the following models: {openai_model_list + claude_model_list + gemini_model_list}")

    if type(prompt) == list:
        prompt_string = ''.join([message.content for message in prompt])
    else:
        prompt_string = prompt

    # logprobs are only available for OpenAI models
    if probabilities and (model in openai_model_list):
        llm = llm.bind(logprobs=True)
    elif probabilities:
        print(
            f"Probabilities are not available for model {model}, please select a model from the following list: {openai_model_list}")

    if dry_run:
        return {'score': -1,
                'error_message': "TEST MSG",
                'prompt': prompt_string}

    # This is the core call to the model
    try:
        response = llm.invoke(prompt)
    except Exception as invocation_error:
        print(f'Error invoking model {model}: {invocation_error}')
        response_dict = {'score': 'ERR',
                         'error_message': invocation_error,
                         'prompt': prompt_string}
        return response_dict

    # This is hardcoded to expect a single score or NA in the response
    # If the desired response changes this will need to be updated
    # Originally this handled a json response but that was removed to make
    # this more robust.
    try:
        score = response.content.strip()
        if not validate_score(score):
            raise ValueError(f'Invalid score: {score}')
        response_dict = {'score': score,
                         'error_message': None,
                         'prompt': prompt_string}

    except Exception as original_error:
        attempt = 1
        while attempt <= parse_retries:
            print(
                f'\nError parsing response from model {model}, retrying attempt {attempt}')
            try:
                response = llm.invoke(prompt)
                score = response.content.strip()
                if not validate_score(score):
                    raise ValueError(f'Invalid score: {score}')
                response_dict = {'score': score,
                                 'error_message': None,
                                 'prompt': prompt_string}
                break
            except:
                attempt += 1
        else:
            print(f'Retries failed with model {model}: {original_error}')
            response_dict = {'score': 'ERR',
                             'error_message': original_error,
                             'prompt': prompt_string}
            return response_dict

    # Extract the probability of the score token from the response metadata
    if probabilities and (model in openai_model_list):
        try:
            score = response_dict['score']
            response_meta_df = pd.DataFrame(
                response.response_metadata["logprobs"]["content"])
            score_metadata = response_meta_df[response_meta_df['token'] == str(
                score)].iloc[0]
            # note this is a little fragile, it will retrieve the probability of the first token in the response
            # that matches the score, which given the template "should" be the score itself, but it's not guaranteed
            prob = np.exp(score_metadata['logprob'])
            response_dict['prob'] = prob
        except Exception as e:
            print(f'Error extracting probabilities from model {model}: {e}')
            response_dict['prob'] = 'ERR'

    return response_dict


def analyze_text_with_batch(prompt_list, model, parse_retries=3, max_retries=7, concurrency=3, dry_run=False):
    """
    Analyzes the given text, given a list of prompts using the specified model.

    Parameters:
    - prompt_list (list): A list of lists, each containing message objects representing the conversation.
    - model (str): The name or ID of the model to use for analysis.
    - parse_retries (int): The number of times to retry parsing the response. Defaults to 3.
    - max_retries (int): The number of times to retry invoking the model. Defaults to 7, which should be enough 
        to handle most TPM rate limits with langchains built in exponential backoff.
    - concurrency (int): The number of concurrent requests to make to the model. Defaults to 3.


    Returns:
    - dict: A dictionary containing the analyzed text generated by the model.

    Checked models:
    - OpenAI models: ['gpt-3.5-turbo', 'gpt-4o', 'gpt-4']
    - Claude models: ['claude-3-5-sonnet-20240620', 'claude-3-opus-20240229', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307']
    - Gemini models: ['gemini-1.5-pro-001']

    """
    # Tested models, this list should be updated as new models are added
    # Each service requires an API key to work, stored as an envorinment variable
    # openai_model_list = ['gpt-3.5-turbo', 'gpt-4o', 'gpt-4']
    # claude_model_list = ['claude-3-5-sonnet-20240620', 'claude-3-opus-20240229',
    #                      'claude-3-sonnet-20240229', 'claude-3-haiku-20240307']
    # gemini_model_list = ['gemini-1.5-pro-001']

    if model in openai_model_list:
        llm = ChatOpenAI(temperature=0, max_tokens=150,
                         model_name=model, max_retries=max_retries)
    elif model in claude_model_list:
        llm = ChatAnthropic(temperature=0, max_tokens=150,
                            model_name=model, max_retries=max_retries)
    elif model in gemini_model_list:
        llm = ChatGoogleGenerativeAI(
            temperature=0, max_tokens=150, model=model, max_retries=max_retries)
    else:
        print("You've selected a model that is not available.")
        print(
            f"Please select from the following models: {openai_model_list + claude_model_list + gemini_model_list}")

    # This is the core call to the model, using batch for concurrency
    # We needed to add concurrency because we hit rate limits with the API
    responses = []
    prompt_batches = [prompt_list[i:i + concurrency]
                      for i in range(0, len(prompt_list), concurrency)]

    if dry_run:
        return [{'score': -1,
                 'error_message': "TEST MSG",
                 'prompt': prompt_list[i]} for i, p in enumerate(prompt_list)]

    for prompt_batch in prompt_batches:
        responses += llm.batch(prompt_batch)

    # This is hardcoded to expect a single score or NA in the response
    # If the desired response changes this will need to be updated
    # Originally this handled a json response but that was removed to make
    # this more robust.
    response_dicts = []
    for i, response in enumerate(responses):
        try:
            score = response.content.strip()
            if not validate_score(score):
                raise ValueError(f'Invalid score: {score}')
            response_dict = {'score': score,
                             'error_message': None, 'prompt': prompt_list[i]}
        except Exception as original_error:
            attempt = 1
            while attempt <= parse_retries:
                print(
                    f'\nError parsing response from model {model}, retrying attempt {attempt}')
                try:
                    response = llm.invoke(prompt_list[i])
                    score = response.content.strip()
                    if not validate_score(score):
                        raise ValueError(f'Invalid score: {score}')
                    response_dict = {
                        'score': score, 'error_message': None, 'prompt': prompt_list[i]}
                    break
                except:
                    attempt += 1
            else:
                print(f'Retries failed with model {model}: {original_error}')
                response_dict = {
                    'score': 'ERR', 'error_message': original_error, 'prompt': prompt_list[i]}
        response_dicts.append(response_dict)

    return response_dicts


def bulk_analyze_text(
        file_list, model_list, issue_list, summarize=True, parse_retries=3, max_retries=7,
        concurrency=3, override_persona_and_encouragement=None, results_file=None,
        output_dir=None, results_file_name="analyze_results.xlsx", if_summary_exists='reuse',
        summary_size=(500, 1000), summary_max_tokens_factor=1.0, dry_run=False
):
    """
    Analyzes a collection of text files using different models and prompts.

    Parameters:
    - file_list (list): A list of file paths containing the texts to analyze.
    - model_list (list): A list of model names to use for analysis. 
    - issue_list (list): A list of issue areas corresponding to each text file.
    - summarize (bool): Whether to summarize the text before analyzing it. Defaults to True.
    - parse_retries (int): The number of times to retry parsing the response. Defaults to 3.
    - max_retries (int): The number of times to retry invoking the model. Defaults to 7, which should be enough
        to handle most TPM rate limits with langchains built in exponential backoff.
    - concurrency (int): The number of concurrent requests to make to the model. Defaults to 3.
    - results_file (str): The path to the Excel file where the results will be saved, if provided,
        output_dir and results_file_name will be ignored.
    - output_dir (str): The path to the output directory where the results will be saved.
    - results_file_name (str): The name of the output file. Defaults to 'analyze_results.xlsx'.
    - if_summary_exists (str): What to do if the summary file already exists. Options are 'overwrite', 'reuse' Defaults to 'reuse'
    - summary_size (tuple, optional): The minimum and maximum size of the final summary. Defaults to (500,1000).
    - summary_max_tokens_factor (float, optional): The max_tokens of LLM will be set to summary_size[1]*summary_max_tokens_factor

    Returns:
    - list: A list of dictionaries containing the analyzed text generated by each model.

    """
    if results_file is None:
        results_file = os.path.join(output_dir, results_file_name)

    overall_results = []
    # Loop through each file, issue area, model and prompt
    for file_name in file_list:
        print('Analyzing file: ', file_name)

        if summarize:
            text = summarize_file(
                file_name, issue_list, output_dir, save_summary=True, if_exists=if_summary_exists,
                summary_size=summary_size, max_tokens_factor=summary_max_tokens_factor
            )
        else:
            with open(file_name, "r", encoding="utf-8") as file:
                text = file.read()

        for issue in issue_list:
            print('-- Analyzing issue: ', issue)
            prompts = get_prompts(
                issue, text, override_persona_and_encouragement)

            for model in model_list:
                print('---- Analyzing with model: ', model)

                results = analyze_text_with_batch(
                    prompts, model, parse_retries=parse_retries, max_retries=max_retries, concurrency=concurrency, dry_run=dry_run)
                results_df = pd.DataFrame(results)
                results_df['issue'] = issue
                results_df['model'] = model
                results_df['file'] = file_name
                results_df['created_at'] = datetime.now()
                results_df = results_df[[
                    'file', 'issue', 'model', 'score', 'error_message', 'prompt', 'created_at']]

                # Writing to Excel as we go to avoid losing data in case of an error
                if os.path.exists(results_file):
                    # Append to existing file
                    with pd.ExcelWriter(results_file, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
                        results_df.to_excel(
                            writer, index=False, header=False, sheet_name='Sheet1',
                            startrow=writer.sheets['Sheet1'].max_row)
                else:
                    # Create a new file
                    with pd.ExcelWriter(results_file, mode='w', engine='openpyxl') as writer:
                        results_df.to_excel(
                            writer, index=False, sheet_name='Sheet1')

                overall_results.append(results_df)

    final_df = pd.concat(overall_results, axis=0)
    final_df = final_df.reset_index(drop=True)
    return final_df


def bulk_analyze_text_few_shot(file_list, model_list, issue_list, results_file=None,
                               output_dir="../data/summaries/",  summarize=True,
                               results_file_name="analyze_few_shot_results.xlsx",
                               parse_retries=3, max_retries=7, if_summary_exists='reuse'):
    """
    Analyzes a collection of text files using different models and prompts.
    It adds a longer few shot prompt based on a calibration set of files. 

    Parameters:
    - file_list (list): A list of file paths containing the texts to analyze.
    - model_list (list): A list of model names to use for analysis. 
    - issue_list (list): A list of issue areas corresponding to each text file.
    - results_file (str): The path to the Excel file where the results will be saved.
    - summarize (bool): Whether to summarize the text before analyzing it. Defaults to True.
    - output_dir (str): The path to the output directory where the summaries will be saved.
    - parse_retries (int): The number of times to retry parsing the response. Defaults to 3.
    - max_retries (int): The number of times to retry invoking the model. Defaults to 7, which should be enough
        to handle most TPM rate limits with langchains built in exponential backoff.
    - if_summary_exists (str): What to do if the summary file already exists. Options are 'overwrite', 'reuse' Defaults to 'reuse'

    Returns:
    - list: A list of dictionaries containing the analyzed text generated by each model.

    """
    if results_file is None:
        results_file = os.path.join(output_dir, results_file_name)

    overall_results = []
    # Loop through each file, issue area, model
    for file_name in file_list:
        print('Analyzing file: ', file_name)

        if summarize:
            text = summarize_file(file_name, issue_list,
                                  output_dir, save_summary=True, if_exists=if_summary_exists)
        else:
            with open(file_name, "r", encoding="utf-8") as file:
                text = file.read()

        for issue in issue_list:
            print('-- Analyzing issue: ', issue)
            prompt = get_few_shot_prompt(issue, text)

            for model in model_list:
                print('---- Analyzing with model: ', model)

                results = analyze_text(
                    prompt, model, parse_retries=parse_retries, max_retries=max_retries)
                results_df = pd.DataFrame([results])
                results_df['issue'] = issue
                results_df['model'] = model
                results_df['file'] = file_name
                results_df['created_at'] = datetime.now()
                results_df = results_df[[
                    'file', 'issue', 'model', 'score', 'error_message', 'prompt', 'created_at']]

                # Writing to Excel as we go to avoid losing data in case of an error
                if os.path.exists(results_file):
                    # Append to existing file
                    with pd.ExcelWriter(results_file, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
                        results_df.to_excel(
                            writer, index=False, header=False, sheet_name='Sheet1',
                            startrow=writer.sheets['Sheet1'].max_row)
                else:
                    # Create a new file
                    with pd.ExcelWriter(results_file, mode='w', engine='openpyxl') as writer:
                        results_df.to_excel(
                            writer, index=False, sheet_name='Sheet1')

                overall_results.append(results_df)

    final_df = pd.concat(overall_results, axis=0)
    final_df = final_df.reset_index(drop=True)
    return final_df


def analyze_dataset(
        summary_dataset, model_list, tag=None, summary_tag=None, results_file=None,
        override_persona_and_encouragement=None, parse_retries=3, max_retries=7,
        use_few_shot=False, output_dir="../data/summaries/", result_file_name="analyze_results.csv",
        dry_run=False, if_exist="reuse"
):
    """
        Analyze the summarizes in a csv file
        This is s function for reproducible run

        Args:
            summary_dataset (str): The path to the csv file of summary texts.
            model_list (str): The names or IDs of the model to use for analysis.
            tag (str): Tag for the run.
            summary_tag (str): Tag for the summary run.
            results_file (str): The path to the Excel file where the results will be saved. If not provided, use output_dir and result_file_name instead.
            output_dir (str): The path to the directory where the results will be stored. Default to "../data/summaries/".
            result_file_name (str): The name of the output file.
            override_persona_and_encouragement (tuple, optional): Use a specific persona and encouragement.
            parse_retries (int): The number of times to retry parsing the response. Defaults to 3.
            max_retries (int): The number of times to retry invoking the model. Defaults to 7, which should be enough
               to handle most TPM rate limits with langchains built in exponential backoff.
            use_few_shot (bool, optional): Whether to use few shot prompt or not. Defaults to False.
            dry_run (bool, optional): Don't invoke the LLM api call. Return a mock response for debug and testing. Defaults to False.
            if_exist (str, optional): Whether to reuse an existing summary result. Defaults to reuse.
        """

    # First create result file or load existing results
    if results_file is None:
        results_file = os.path.join(output_dir, result_file_name)
    columns = [
        'filename', 'issue', 'summary_model', 'tag', 'model', 'score',
        'error_message', 'prompt', 'created_at'
    ]
    if os.path.exists(results_file):
        df_existing_results = pd.read_csv(results_file)
    else:
        df_existing_results = pd.DataFrame(columns=columns)
        df_existing_results.to_csv(results_file, index=False)

    # Load all summaries to be analyzed
    df_summaries = pd.read_csv(summary_dataset)
    if summary_tag is not None:
        df_summaries = df_summaries[df_summaries['tag'] == summary_tag]

    for summary_record in df_summaries.itertuples():
        print('Analyzing: ', summary_record.filename)
        for issue in summary_record.issues.split("-"):
            print('-- Analyzing issue: ', issue)
            for model in model_list:
                print('---- Analyzing with model: ', model)
                existing_record = df_existing_results[
                    (df_existing_results["filename"]==summary_record.filename)
                    &(df_existing_results["issue"]==issue)
                    &(df_existing_results["summary_model"]==summary_record.summary_model)
                    &(df_existing_results['tag'] == tag)
                    &(df_existing_results['model'] == model)
                ]
                if existing_record.shape[0] > 0:
                    if if_exist == "reuse":
                        print(f"Skip: {tag} {summary_record.summary_model} summarizer")
                    elif existing_record.shape[0] > 1:
                        warnings.warn(f"Record has multiple existing summaries: {summary_record.filename}")
                    else:
                        df_existing_results = df_existing_results.drop(existing_record.index)

                if use_few_shot:
                    prompt = get_few_shot_prompt(issue, summary_record.summary)
                    results = analyze_text(
                        prompt, model, parse_retries=parse_retries, max_retries=max_retries, dry_run=dry_run
                    )
                    results_df = pd.DataFrame([results])
                else:
                    prompt = get_prompts(issue, summary_record.summary, override_persona_and_encouragement)
                    results = analyze_text_with_batch(
                        prompt, model, parse_retries=parse_retries, max_retries=max_retries, dry_run=dry_run
                    )
                    results_df = pd.DataFrame(results)
                results_df['filename'] = summary_record.filename
                results_df['issue'] = issue
                results_df['model'] = model
                results_df['summary_model'] = summary_record.summary_model
                results_df['tag'] = tag
                results_df['created_at'] = datetime.now()
                results_df = results_df[[
                    'filename', 'issue', 'summary_model', 'tag', 'model', 'score', 'error_message', 'prompt',
                    'created_at']]
                if existing_record.shape[0] > 0:
                    df_new = pd.concat([df_existing_results, results_df])
                    df_new.to_csv(results_file, mode="w", index=False)
                else:
                    results_df.to_csv(results_file, mode="a", index=False, header=False)
