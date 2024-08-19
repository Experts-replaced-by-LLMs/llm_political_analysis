# Note this still isn't used in the main flow yet, just an option for the future.
import sqlite3
from datetime import datetime, timezone
from google.cloud import storage

import pandas as pd


def store_results(results_df, db_path='../data/results/manifesto_analysis_results.db', table_name='results'):
    """
    Stores the results of the analysis in a SQLite database.

    Args:
        results_df (pd.DataFrame): The DataFrame containing the results of the analysis.
        db_path (str): The path to the SQLite database file. 

    Returns:
        None
    """
    conn = sqlite3.connect(db_path)
    results_df['datetime_utc_added'] = datetime.now(timezone.utc)
    results_df.to_sql(table_name, conn, if_exists='append', index=False, method='multi')
    print(f'{len(results_df)} results stored successfully in the database.')


def get_results(db_path='../data/results/manifesto_analysis_results.db', table_name='results'):
    """
    Retrieves the results of the analysis from a SQLite database.

    Args:
        db_path (str): The path to the SQLite database file. 

    Returns:
        pd.DataFrame: The DataFrame containing the results of the analysis.
    """
    conn = sqlite3.connect(db_path)
    results_df = pd.read_sql(f'SELECT * FROM {table_name}', conn)
    return results_df


def read_gcs_file(
        blob_name,
        bucket_name="llms-as-experts",
        project="llms-as-experts",
        encoding="utf-8"
):
    """
    Read file content from Google Cloud Storage

    Args:
        blob_name (str): The URI of the file, excluding bucket name.
        bucket_name (str): The name of the bucket.
        project (str): The name of the project.
        encoding (str): File encoding.

    """
    storage_client = storage.Client(project=project)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    with blob.open("r", encoding=encoding) as f:
        return f.read()


def list_gcs_folder(
        prefix,
        bucket_name="llms-as-experts",
        project="llms-as-experts",
        delimiter=None
):
    storage_client = storage.Client(project=project)
    return [
        str(blob.name)
        for blob in storage_client.list_blobs(bucket_name, prefix=prefix, delimiter=delimiter)
    ]


def download_gcs_file(
        output_filename,
        blob_name,
        bucket_name="llms-as-experts",
        project="llms-as-experts",
):
    storage_client = storage.Client(project=project)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(
        output_filename
    )
