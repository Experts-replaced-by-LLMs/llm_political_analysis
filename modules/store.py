# Note this still isn't used in the main flow yet, just an option for the future.
import sqlite3
from datetime import datetime, timezone

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