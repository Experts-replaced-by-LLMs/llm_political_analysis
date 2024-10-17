import os.path
import re
import time
import pickle
import warnings
from uuid import uuid4

import anthropic
import pandas as pd

from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage

from llm_political_analysis.modules import openai_model_list, claude_model_list, gemini_model_list, ollama_model_list, \
    per_minute_token_limit
from llm_political_analysis.modules.prompts import load_prompts


def summarize_text(
        text, issue_areas, model="gpt-4o", chunk_size=100000, overlap=2500, summary_size=(500, 1000),
        max_tokens_factor=1.0, prompt_version=None, return_log=False, debug=False, dry_run=False
):
    """
    Summarizes the given text based on the specified issue areas using a language model.
    This approach creates all the summaries in one pass to avoid repeating the summarization process.

    Args:
        text (str): The text to be summarized.
        issue_areas (list): The issue areas related to the text.
        model (str, optional): The name of the language model to be used for summarization. Defaults to "gpt-4o".
        chunk_size (int, optional): The size of each chunk to split the text into. Defaults to 100000. Set to 0 to disable chunk.
        overlap (int, optional): The overlap between consecutive chunks. Defaults to 2500.
        summary_size (tuple, optional): The minimum and maximum size of the final summary. Defaults to (500,1000).
        max_tokens_factor (float, optional): The max_tokens of LLM will be set to summary_size[1]*max_tokens_factor
        prompt_version (str, optional): The prompt version to be used.
        return_log (bool, optional): Whether to return log information. Defaults to False.
        debug (bool, optional): Should debug information be printed. Defaults to False.
        dry_run (bool, optional): Don't invoke the LLM api call. Return a mock response for debug and testing. Defaults to False.

    Returns:
        str: The final summary of the text.
    """
    # Handle single issues gracefully
    if isinstance(issue_areas, str):
        issue_areas = [issue_areas]

    if chunk_size > 0:
        if chunk_size < 1:
            chunk_size = int(len(text)*chunk_size)
        # Split the text into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
        chunks = text_splitter.split_text(text)
    else:
        chunks = [text]

    # Craft the prompt templates
    # Load summarize prompts
    summarize_prompts = load_prompts("summarize", version=prompt_version)
    system_template_string = summarize_prompts["system_template_string"]
    human_template_string = summarize_prompts["human_template_string"]
    policy_areas = summarize_prompts["policy_areas"]

    system_template = PromptTemplate(template=system_template_string)
    human_template = PromptTemplate(template=human_template_string)

    issue_area_dict = {issue: policy_areas.get(issue, 'general policy issues') for issue in issue_areas}
    issue_area_descriptions = [f"{issue}: {description}" for issue, description in issue_area_dict.items()]
    issue_list_string = "\n".join([f"{i + 1}. {area}" for i, area in enumerate(issue_area_descriptions)])

    # Setup the LLM
    # llm = ChatOpenAI(temperature=0, max_tokens=summary_size[1] * max_tokens_factor, model_name=model)
    max_tokens = summary_size[1] * max_tokens_factor
    if model in openai_model_list:
        llm = ChatOpenAI(temperature=0, max_tokens=max_tokens, model_name=model)
    elif model in claude_model_list:
        llm = ChatAnthropic(temperature=0, max_tokens=max_tokens, model_name=model)
    elif model in gemini_model_list:
        llm = ChatGoogleGenerativeAI(temperature=0, max_tokens=max_tokens, model=model)
    elif model in ollama_model_list:
        ollama_url = os.getenv("OLLAMA_URL")
        if not ollama_url:
            raise ValueError("To use ollama model, set OLLAMA_URL variable.")
        llm = ChatOllama(temperature=0, num_predict=max_tokens, model=model, base_url=ollama_url)
    else:
        raise Exception(
            f"You've selected a model that is not available.\nPlease select from the following models: {openai_model_list + claude_model_list + gemini_model_list}"
        )

    logs = []
    # Summarize each chunk
    summaries = []
    tokens_used = 0
    # token_limit = 800000
    token_limit = per_minute_token_limit.get(model, 800000)
    start_time = time.time()

    print(f"Using {model} for summarization. Token limit: {token_limit}/min")

    for chunk in chunks:
        # This handling is needed for the input rate limit, for gpt-4o thats 30k tokens per minute for tier 1, 800k tokens per minute for tier 3
        if tokens_used + len(chunk) / 4 > token_limit:

            elapsed_time = time.time() - start_time
            time_to_wait = 60 - elapsed_time
            if time_to_wait > 0:
                print(f'Waiting for {time_to_wait:.0f} seconds to avoid token limit. Tokens used: {tokens_used}')
                time.sleep(time_to_wait)
            tokens_used = 0
            start_time = time.time()

        summarize_prompt = [
            SystemMessage(
                content=system_template.format(
                    issue_areas=issue_list_string,
                    min_size=summary_size[0],
                    max_size=summary_size[1]
                )),
            HumanMessage(
                content=human_template.format(text=chunk)
            )
        ]

        if debug:
            print('Prompt:', summarize_prompt)

        if dry_run:
            return text[:100], [(text[100:200], text[-100:])]

        try:
            summary = llm.invoke(summarize_prompt)
            logs.append((summarize_prompt, summary))
        except anthropic.RateLimitError:
            print("Anthropic rate limit exceeded. Waiting for 1 minute ...")
            time.sleep(60)
            print("Retry Anthropic invoke.")
            summary = llm.invoke(summarize_prompt)
            logs.append((summarize_prompt, summary))

        summaries.append(summary.content)
        try:
            tokens_used += summary.response_metadata['token_usage']['prompt_tokens']
        except:
            # Ollama model does not have token usage. Nothing needs to be done.
            pass
        print(f'Summarized so far: {len(summaries)} out of {len(chunks)} chunks', end='\r')
    print('\n', end='\r')

    # Combine all summaries into one final summary
    if len(summaries) > 1:
        print('Combining summaries into one final summary')
        final_summaries = " ".join(summaries)
        final_summarize_prompt = [
            SystemMessage(
                content=system_template.format(
                    issue_areas=issue_list_string,
                    min_size=summary_size[0],
                    max_size=summary_size[1]
                )),
            HumanMessage(
                content=human_template.format(text=final_summaries)
            )
        ]
        final_summary = llm.invoke(final_summarize_prompt).content
        logs.append((final_summarize_prompt, final_summary))
    else:
        final_summary = summaries[0]

    print(f'Final summary length: {len(final_summary)} characters \n')

    if return_log:
        return final_summary, logs
    return final_summary


def summarize_file(file_path, issue_areas, output_dir="../data/summaries/", model="gpt-4o",
                   chunk_size=100000, overlap=2500, summary_size=(500, 1000), if_exists='overwrite',
                   save_summary=True, max_tokens_factor=1.0, prompt_version=None, save_log=False, debug=False):
    """
    Summarizes the text in the given file based on the specified issue area using a language model.

    Args:
        file_path (str): The path to the file containing the text to be summarized.
        issue_areas (list): The issue areas related to the text.
        output_dir (str): The path to the directory where the results will be stored. Default to "../data/summaries/".
        model (str, optional): The name of the language model to be used for summarization. Defaults to "gpt-4o".
        chunk_size (int, optional): The size of each chunk to split the text into. Defaults to 100000.
        overlap (int, optional): The overlap between consecutive chunks. Defaults to 2500.
        summary_size (tuple, optional): The minimum and maximum size of the final summary. Defaults to (500,1000).
        if_exists (str, optional): What to do if the summary file already exists. Options are 'overwrite', 'reuse' Defaults to 'overwrite'
        save_summary (bool, optional): Should the summary be saved to a file. Defaults to True.
        max_tokens_factor (float, optional): The max_tokens of LLM will be set to summary_size[1]*max_tokens_factor
        prompt_version (str, optional): The prompt version to be used.
        save_log (bool, optional): Should the log information be saved to a file. Defaults to False.
        debug (bool, optional): Should debug information be printed. Defaults to False.

    Returns:
        str: The final summary of the text.
    """
    # Handle single issues gracefully
    if isinstance(issue_areas, str):
        issue_areas = [issue_areas]

    # Constructs the output file name
    input_filename, _ = os.path.splitext(os.path.basename(file_path))

    if summary_size[1] == 1000:
        length = 'standard'
    elif summary_size[1] < 1000:
        length = 'short'
    else:
        length = 'long'

    # Format model name for output
    model_name = re.sub(r'[-_.:]', '', model)
    if len(issue_areas) > 1:
        summary_file_name = os.path.join(
            output_dir, f"summary_{model_name}_{length}_multi_issue__{input_filename}.txt"
        )
    else:
        summary_file_name = os.path.join(
            output_dir, f"summary_{model_name}_{length}_{issue_areas[0]}__{input_filename}.txt"
        )

    # Check if the summary file already exists, and reuses it if requested, exiting early
    if if_exists == 'reuse':
        if os.path.exists(summary_file_name):
            print(f"Summary file {summary_file_name} already exists. Reusing the existing summary.")
            with open(summary_file_name, "r", encoding="utf-8") as file:
                return file.read()

    # Main summarization process
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
    summary = summarize_text(
        text, issue_areas, model=model, chunk_size=chunk_size, overlap=overlap, summary_size=summary_size, debug=debug,
        max_tokens_factor=max_tokens_factor, prompt_version=prompt_version, return_log=save_log
    )
    if save_log:
        summary, logs = summary
        log_file_name = os.path.join(
            output_dir, f"summary_log_{model_name}_{length}_{issue_areas[0]}__{input_filename}.pkl"
        )
        with open(log_file_name, 'wb') as f:
            pickle.dump(logs, f)
    if save_summary:
        print(f"Saving summary to {summary_file_name}")
        with open(summary_file_name, "w", encoding="utf-8") as file:
            file.write(summary)

    return summary


def summarize_dataset(
        dataset_filepath, group, issue_areas, output_dir="../data/summaries/", output_filename="summaries.csv",
        model="gpt-4o", try_no_chunk=False, chunk_size=100000, overlap=2500, summary_size=(300, 400), if_exists='reuse',
        max_tokens_factor=2.0, prompt_version=None, save_log=False, tag=None, debug=False, dry_run=False
):
    """
    Summarizes the text in the given dataset file based on the specified group and issue area using a language model.
    Save the summaries into a csv file.
    This is s function for reproducible run

    Args:
        dataset_filepath: The path to the dataset file of original manifesto documents.
        group: Subset of the dataset. One of [prototyping, production, calibration, test, coalition, translation]
        issue_areas (list): The issue areas related to the text.
        output_dir: (str): The path to the directory where the results will be stored. Default to "../data/summaries/".
        output_filename (str): The name of the final summary file.
        model (str, optional): The name of the language model to be used for summarization. Defaults to "gpt-4o".
        try_no_chunk (bool, optional): Try to summarize without chunking first. Fall back to chunking when exception raised.
        chunk_size (int, optional): The size of each chunk to split the text into. Defaults to 100000.
        overlap (int, optional): The overlap between consecutive chunks. Defaults to 2500.
        summary_size (tuple, optional): The minimum and maximum size of the final summary. Defaults to (500,1000).
        if_exists (str, optional): What to do if the summary already exists. Options are 'overwrite', 'reuse' Defaults to 'reuse'
        max_tokens_factor (float, optional): The max_tokens of LLM will be set to summary_size[1]*max_tokens_factor
        prompt_version (str, optional): The prompt version to be used.
        save_log (bool, optional): Should the log information be saved to a file. Defaults to False.
        tag: To continue a break run, always pass in a unique tag
        debug (bool, optional): Should debug information be printed. Defaults to False.
        dry_run (bool, optional): Don't invoke the LLM api call. Return a mock response for debug and testing. Defaults to False.

    """
    if tag is None:
        tag = uuid4().hex

    columns = ["filename", "issues", "summary_model", "summary", "timestamp", "tag"]

    if isinstance(issue_areas, str):
        issue_areas = [issue_areas]

    issue_areas_string = "-".join(issue_areas)

    df_manifesto_dataset = pd.read_csv(dataset_filepath)
    df_manifesto_dataset = df_manifesto_dataset[df_manifesto_dataset[group] == 1]

    result_filepath = os.path.join(output_dir, output_filename)

    for _, record in df_manifesto_dataset.iterrows():

        if os.path.exists(result_filepath):
            df_existing_summaries = pd.read_csv(result_filepath)
        else:
            df_existing_summaries = pd.DataFrame(columns=columns)
            df_existing_summaries.to_csv(result_filepath, index=False)
        new_summaries = []

        existing_rec = df_existing_summaries[
            (df_existing_summaries["filename"]==record["filename"])
            &(df_existing_summaries["issues"]==issue_areas_string)
            &(df_existing_summaries["summary_model"]==model)
            &(df_existing_summaries["tag"]==tag)
            ]
        if existing_rec.shape[0] > 0:
            if if_exists == "reuse":
                print(f"Skip summary: [{issue_areas_string} {tag}] {record['filename']}")
                continue
            elif existing_rec.shape[0] > 1:
                warnings.warn(f"Record has multiple existing summaries: {record['filename']}")
            else:
                # Remove previous one
                df_existing_summaries = df_existing_summaries.drop(existing_rec.index)

        print(f"--- Summarizing {record['filename']} ---")

        if try_no_chunk:
            try:
                print("Trying summarize without chunk ...")
                summary = summarize_text(
                    record["text"], issue_areas, model=model, chunk_size=0, overlap=0, summary_size=summary_size,
                    max_tokens_factor=max_tokens_factor, prompt_version=prompt_version, return_log=save_log,
                    debug=debug, dry_run=dry_run
                )
            except Exception as e:
                if chunk_size>0:
                    print("Cannot summarize entire document. Trying chunk summarization...")
                    summary = summarize_text(
                        record["text"], issue_areas, model=model, chunk_size=chunk_size, overlap=overlap,
                        summary_size=summary_size,
                        max_tokens_factor=max_tokens_factor, prompt_version=prompt_version, return_log=save_log,
                        debug=debug, dry_run=dry_run
                    )
                else:
                    raise e
        else:
            summary = summarize_text(
                record["text"], issue_areas, model=model, chunk_size=chunk_size, overlap=overlap,
                summary_size=summary_size,
                max_tokens_factor=max_tokens_factor, prompt_version=prompt_version, return_log=save_log,
                debug=debug, dry_run=dry_run
            )

        if save_log:
            log_paths = os.path.join(output_dir, "logs")
            if not os.path.exists(log_paths):
                os.makedirs(log_paths)
            summary, logs = summary
            model_name_string = re.sub(r'[-_.:]', '', model)
            log_file_name = os.path.join(
                log_paths, f"summary_log_{issue_areas_string}_{tag}_{model_name_string}__{os.path.splitext(record['filename'])[0]}.pkl"
            )
            with open(log_file_name, 'wb') as f:
                pickle.dump(logs, f)

        new_summaries.append([record["filename"], issue_areas_string, model, summary, int(time.time()), tag])
        if existing_rec.shape[0] > 0:
            df_new = pd.concat([df_existing_summaries, pd.DataFrame(new_summaries)])
            df_new.to_csv(result_filepath, mode="w", index=False)
        else:
            df_new = pd.DataFrame(new_summaries, columns=columns)
            df_new.to_csv(result_filepath, mode="a", index=False, header=False)
