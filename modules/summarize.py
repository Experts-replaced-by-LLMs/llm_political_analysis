from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate


def summarize_text(text, issue_area, model="gpt-4o", chunk_size=5000, overlap=250):
    """
    Summarizes the given text based on the specified issue area using a language model.

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
                    formatting. ''',
                ),
                ("human", "{input}"),
            ]
        )
        llm=ChatOpenAI(temperature=0, max_tokens=4000, model_name=model)
        summarize_chain = prompt | llm
        summary = summarize_chain.invoke({"input": chunk, "issue_area": issue_area})
        summaries.append(summary.content)
        print(f'Summarized so far: {len(summaries)} out of {len(chunks)} chunks', end='\r')
    print('\n')

    # Combine all summaries into one final summary
    final_summaries = " ".join(summaries)
    final_summary = summarize_chain.invoke({"input": final_summaries, "issue_area": issue_area}).content

    # Check if the final summary is still too long, if so, summarize again
    if len(final_summary) > 5000:
        print(f'Condensing final summary since it is {len(final_summary)} characters')
        final_summary =  summarize_chain.invoke({"input": final_summary, "issue_area": issue_area}).content

    return final_summary
