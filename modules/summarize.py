import os.path
import time

from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage

from .prompts import policy_areas


def summarize_text(text, issue_areas, model="gpt-4o", chunk_size=100000, overlap=2500, summary_size=(500,1000)):
    """
    Summarizes the given text based on the specified issue areas using a language model.
    This approach creates all the summaries in one pass to avoid repeating the summarization process.

    Args:
        text (str): The text to be summarized.
        issue_areas (list): The issue areas related to the text.
        model (str, optional): The name of the language model to be used for summarization. Defaults to "gpt-4o".
        chunk_size (int, optional): The size of each chunk to split the text into. Defaults to 100000.
        overlap (int, optional): The overlap between consecutive chunks. Defaults to 2500.
        summary_size (tuple, optional): The minimum and maximum size of the final summary. Defaults to (500,1000).

    Returns:
        str: The final summary of the text.
    """
    # Handle single issues gracefully
    if isinstance(issue_areas, str):
        issue_areas = [issue_areas]

    # Split the text into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = text_splitter.split_text(text)

    # Craft the prompt templates
    system_template_string = '''
        You are an expert political analyst. Please summarize the following political manifesto. You should detect the initial language and output the summaries in English.
        The summary should highlight key points and policy positions specifically related to the following topics, as these will be evaluated later:

        {issue_areas}

        Aim for a concise summary of around {min_size}-{max_size} words that covers these key policy areas, and be sure they are all present on the original text.
        Give bullet points for each area, and format the output as plaintext. 
        '''
    system_template = PromptTemplate(template=system_template_string)
    human_template = PromptTemplate(template='Please summarize the following text:\n{text}')

    issue_area_dict = {issue:policy_areas.get(issue, 'general policy issues') for issue in issue_areas}
    issue_area_descriptions = [f"{issue}: {description}" for issue, description in issue_area_dict.items()]
    issue_list_string = "\n".join([f"{i+1}. {area}" for i, area in enumerate(issue_area_descriptions)])


    # Setup the LLM
    llm=ChatOpenAI(temperature=0, max_tokens=summary_size[1], model_name=model)

    # Summarize each chunk
    summaries = []
    tokens_used = 0
    token_limit = 800000
    start_time = time.time()

    for chunk in chunks:
        # This handling is needed for the input rate limit, for gpt-4o thats 30k tokens per minute for tier 1, 800k tokens per minute for tier 3
        if tokens_used + len(chunk)/4 > token_limit:
            
            elapsed_time = time.time() - start_time
            time_to_wait = 60 - elapsed_time
            if time_to_wait > 0:
                print(f'Waiting for {time_to_wait:.0f} seconds to avoid token limit')
                time.sleep(time_to_wait)
            tokens_used = 0
            start_time = time.time()

        summarize_prompt = [SystemMessage(content=system_template.format(issue_areas=issue_list_string, min_size=summary_size[0], max_size=summary_size[1])),
                        HumanMessage(content=human_template.format(text=chunk))] 
        summary = llm.invoke(summarize_prompt)
        summaries.append(summary.content)
        tokens_used += summary.response_metadata['token_usage']['prompt_tokens']
        print(f'Summarized so far: {len(summaries)} out of {len(chunks)} chunks', end='\r')
    print('\n', end='\r')

    # Combine all summaries into one final summary
    if len(summaries) > 1:
        print('Combining summaries into one final summary')
        final_summaries = " ".join(summaries)
        final_summarize_prompt = [SystemMessage(content=system_template.format(issue_areas=issue_list_string)),
                        HumanMessage(content=human_template.format(text=final_summaries))] 
        final_summary = llm.invoke(final_summarize_prompt).content
    else:
        final_summary = summaries[0]

    print(f'Final summary length: {len(final_summary)} characters \n')
    return final_summary

def summarize_file(file_path, issue_areas, output_dir="../data/summaries/", model="gpt-4o",
                   chunk_size=100000, overlap=2500, summary_size=(500,1000), if_exists='overwrite', save_summary=True):
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

    if len(issue_areas) > 1:
        summary_file_name = os.path.join(output_dir, f"summary_{length}_multi_issue_{input_filename}.txt")
    else:
        summary_file_name = os.path.join(output_dir, f"summary_{length}_{issue_areas[0]}_{input_filename}.txt")

    # Check if the summary file already exists, and reuses it if requested, exiting early
    if if_exists == 'reuse':
        if os.path.exists(summary_file_name):
            print(f"Summary file {summary_file_name} already exists. Reusing the existing summary.")
            with open(summary_file_name, "r", encoding="utf-8") as file:
                return file.read()

    # Main summarization process
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
    summary = summarize_text(text, issue_areas, model=model, chunk_size=chunk_size, overlap=overlap, summary_size=summary_size)
    if save_summary:
        print(f"Saving summary to {summary_file_name}")
        with open(summary_file_name, "w", encoding="utf-8") as file:
            file.write(summary)

    return summary
