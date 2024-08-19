import os
from math import floor
from sys import getsizeof

from google.cloud import translate
from langchain_core.documents import Document
from langchain_google_community import GoogleTranslateTransformer


def truncate_text(text, limit=100000):
    """
    A simple function to truncate input text to meet the length requirements of Google translation's language detection API.

    Args:
        text (str): Text to be truncated.
        limit (int): Maximum number of bytes allowed.
    """
    factor = limit / getsizeof(text.encode("utf-8"))
    truncated = text[:floor(len(text) * factor)]
    return truncated


def detect_language(
        text: str,
        project_id: str = "llms-as-experts",
        location: str = "global",
):
    """
    Detect text language.
    Args:
        text (str): Text to be detected.
        project_id (str): The ID of the project that owns the destination bucket.
        location (str): Translation service location. Default to "global".
    """

    client = translate.TranslationServiceClient()
    parent = f"projects/{project_id}/locations/{location}"
    content = truncate_text(text)
    response = client.detect_language(
        content=content,
        parent=parent,
        mime_type="text/plain",
    )
    try:
        return response.languages[0].language_code
    except Exception:
        return None


def batch_translate_text(
        input_blob_name: str,
        output_blob_name: str,
        source_language_code: str,
        bucket_name: str = "llms-as-experts",
        project_id: str = "llms-as-experts",
        location: str = "global",
        timeout: int = 300,
        target_language_code: str = "en",
) -> translate.TranslateTextResponse:
    """
    Translates a file on GCS and stores the result in a GCS location.
    This uses Google cloud translation's batch_translate_text API, which allow long text translation.
    The source file has to be stored in Google Cloud Storage.

    Args:
        input_blob_name (str): The input file of the texts to be translated. Do not include bucket name.
        output_blob_name (str): The output folder of the translated texts. Do not include bucket name.
        source_language_code (str): The source language codes to translate.
        bucket_name (str): The name of the bucket where the translated texts will be stored.
        project_id (str): The ID of the project that owns the destination bucket.
        location (str): Translation service location. Default to "global".
        timeout (int): The timeout for this translation operation.
        target_language_code (str): The target language codes to translate to. Defaults to "en".

    Returns:
        The translated text.
    """

    input_uri = f"gs://{bucket_name}/{input_blob_name}"
    gcs_source = {"input_uri": input_uri}
    input_configs_element = {
        "gcs_source": gcs_source,
        "mime_type": "text/plain",  # Can be "text/plain" or "text/html".
    }

    output_uri = f"gs://{bucket_name}/{output_blob_name}/"
    gcs_destination = {"output_uri_prefix": output_uri}
    output_config = {"gcs_destination": gcs_destination}

    client = translate.TranslationServiceClient()
    parent = f"projects/{project_id}/locations/{location}"

    # Supported language codes: https://cloud.google.com/translate/docs/languages
    operation = client.batch_translate_text(
        request={
            "parent": parent,
            "source_language_code": source_language_code,
            "target_language_codes": [target_language_code],
            "input_configs": [input_configs_element],
            "output_config": output_config,
        }
    )

    print("Waiting for operation to complete...")
    response = operation.result(timeout)

    print(f"Total Characters: {response.total_characters}")
    print(f"Translated Characters: {response.translated_characters}")

    return response


def translate_text(
        text: str, target_language_code="en", source_language_code=None, project_id="llms-as-experts",
        model_id="general/nmt", location="global"
) -> Document:
    """
    Translate the given text using LangChain's GoogleTranslateTransformer with Google cloud translation as backend.
    This function can only support a limited length of text. Details see: https://cloud.google.com/translate/quotas

    Args:
        text (str): Text to be translated.
        target_language_code (str): The target language codes to translate to. Defaults to "en".
        source_language_code (str, optional): The source language codes to translate. If None, will be detected automatically.
        project_id (str): The ID of the project that owns the destination bucket.
        model_id (str): The model to use. One of [general/nmt, general/translation-llm].
        location (str): The location of model.
    """
    translator = GoogleTranslateTransformer(project_id=project_id, model_id=model_id, location=location)
    documents = [Document(page_content=text)]
    translated_documents = translator.transform_documents(
        documents, target_language_code=target_language_code, source_language_code=source_language_code
    )
    return translated_documents[0]


def translate_file(
        filepath, output_dir="../data/translations/", save_translation=True,
        target_language_code="en", source_language_code=None, project_id="llms-as-experts",
        model_id="general/nmt", location="global"
) -> Document:

    """
    Translate the given file using LangChain's GoogleTranslateTransformer with Google cloud translation as backend.
    This function can only support a limited length of text. Details see: https://cloud.google.com/translate/quotas

    Args:
        filepath (str): The path to the file containing the text to be translated.
        output_dir (str): The path to the directory where the results will be stored. Default to "../data/translations/".
        target_language_code (str): The target language codes to translate to. Defaults to "en".
        source_language_code (str, optional): The source language codes to translate. If None, will be detected automatically.
        project_id (str): The ID of the project that owns the destination bucket.
        model_id (str): The model to use. One of [general/nmt, general/translation-llm].
        location (str): The location of model.
        save_translation (bool): Whether to save the translated text as a file.
    """

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    translated_text = translate_text(
        text, target_language_code=target_language_code, source_language_code=source_language_code,
        project_id=project_id, model_id=model_id, location=location
    )

    if not source_language_code:
        # Check detected source language code
        source_language_code = translated_text.metadata["detected_language_code"]

    if save_translation:
        input_filename, _ = os.path.splitext(os.path.basename(filepath))
        translation_file_name = os.path.join(
            output_dir, f"{input_filename}_translation_{source_language_code}_to_{target_language_code}.txt"
        )
        with open(translation_file_name, "w", encoding="utf-8") as file:
            file.write(translated_text.page_content)

    return translated_text
