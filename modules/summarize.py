import os.path
import time

from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema import HumanMessage, SystemMessage

from .prompts import policy_areas


def summarize_text(text, issue, model="gpt-4o", chunk_size=350000, overlap=2500):
    """
    Summarizes the given text based on the specified issue area using a language model.
    This was the original approach to summarization, but it is not as efficient as the summarize_text_all_issues function.

    Args:
        text (str): The text to be summarized.
        issue_area (str): The issue area related to the text.
        model (str, optional): The name of the language model to be used for summarization. Defaults to "gpt-4o".
        chunk_size (int, optional): The size of each chunk to split the text into. Defaults to 5000.
        overlap (int, optional): The overlap between consecutive chunks. Defaults to 250.

    Returns:
        str: The final summary of the text.
    """
    # Split the text into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = text_splitter.split_text(text)
    
    # Summarize each chunk
    summaries = []
    for chunk in chunks:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    '''You are a helpful text summarizer. You should detect the initial language and output the summaries in English 
                    you want to retain important policy information related to {issue_area}. Return the summaries as a list with minimal
                    formatting. The summaries will be analyzed to determine the party's position on the policy area, on a
                    spectrum so be sure to select the most important policy points''',
                ),
                ("human", "{input}"),
            ]
        )
        llm=ChatOpenAI(temperature=0, max_tokens=1000, model_name=model)
        summarize_chain = prompt | llm
        summary = summarize_chain.invoke({"input": chunk, "issue_area": policy_areas.get(issue, 'general policy issues')})
        summaries.append(summary.content)
        print(f'Summarized so far: {len(summaries)} out of {len(chunks)} chunks', end='\r')
    print('\n', end='\r')

    # Combine all summaries into one final summary
    if len(summaries) > 1:
        final_summaries = " ".join(summaries)
        final_summary = summarize_chain.invoke({"input": final_summaries, "issue_area": policy_areas.get(issue, 'general policy issues')}).content
    else:
        final_summary = summaries[0]

    # Check if the final summary is still too long, if so, summarize again
    if len(final_summary) > 5000:
        print(f'Condensing final summary since it is {len(final_summary)} characters')
        final_summary =  summarize_chain.invoke({"input": final_summary, "issue_area": policy_areas.get(issue, 'general policy issues')}).content

    print(f'Final summary length: {len(final_summary)} characters \n')
    return final_summary

def summarize_text_all_issues(text, issue_areas, model="gpt-4o", chunk_size=100000, overlap=2500):
    """
    Summarizes the given text based on the specified issue areas using a language model.
    This approach creates all the summaries in one pass to avoid repeating the summarization process.

    Args:
        text (str): The text to be summarized.
        issue_areas (list): The issue areas related to the text.
        model (str, optional): The name of the language model to be used for summarization. Defaults to "gpt-4o".
        chunk_size (int, optional): The size of each chunk to split the text into. Defaults to 100000.
        overlap (int, optional): The overlap between consecutive chunks. Defaults to 2500.

    Returns:
        str: The final summary of the text.
    """
    # Split the text into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = text_splitter.split_text(text)

    # Craft the prompt templates
    system_template_string = '''
        You are an expert political analyst. Please summarize the following political manifesto. You should detect the initial language and output the summaries in English.
        The summary should highlight key points and policy positions specifically related to the following topics, as these will be evaluated later:

        {issue_areas}

        Aim for a concise summary of around 500-1000 words that covers these key policy areas. Giving bullet points for each area. 
        '''
    system_template = PromptTemplate(template=system_template_string)
    issue_area_descriptions = [policy_areas.get(issue, 'general policy issues') for issue in issue_areas]
    issue_list_string = "\n".join([f"{i+1}. {area}" for i, area in enumerate(issue_area_descriptions)])
    human_template = PromptTemplate(template='Please summarize the following text:\n{text}')

    # Setup the LLM
    llm=ChatOpenAI(temperature=0, max_tokens=1000, model_name=model)

    # Summarize each chunk
    summaries = []
    tokens_used = 0
    token_limit = 30000
    start_time = time.time()

    for chunk in chunks:
        # This handling is needed for the input rate limit, for gpt-4o thats 30k tokens per minute
        if tokens_used + len(chunk)/4 > token_limit:
            
            elapsed_time = time.time() - start_time
            time_to_wait = 60 - elapsed_time
            if time_to_wait > 0:
                print(f'Waiting for {time_to_wait:.0f} seconds to avoid token limit')
                time.sleep(time_to_wait)
            tokens_used = 0
            start_time = time.time()

        summarize_prompt = [SystemMessage(content=system_template.format(issue_areas=issue_list_string)),
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


def summarize_file(file_path, issue_areas, output_dir, model="gpt-4o", chunk_size=100000, overlap=2500, save_summary=True):
    """
    Summarizes the text in the given file based on the specified issue area using a language model.

    Args:
        file_path (str): The path to the file containing the text to be summarized.
        issue_areas (list): The issue areas related to the text.
        model (str, optional): The name of the language model to be used for summarization. Defaults to "gpt-4o".
        chunk_size (int, optional): The size of each chunk to split the text into. Defaults to 100000.
        overlap (int, optional): The overlap between consecutive chunks. Defaults to 2500.

    Returns:
        str: The final summary of the text.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    summary = summarize_text_all_issues(text, issue_areas, model, chunk_size, overlap)

    if save_summary:
        input_filename, _ = os.path.splitext(os.path.basename(file_path))
        summary_file_name = os.path.join(output_dir, f"{input_filename}_summary.txt")
        print(f"Saving summary to {summary_file_name}")
        with open(summary_file_name, "w", encoding="utf-8") as file:
            file.write(summary)

    return summary
