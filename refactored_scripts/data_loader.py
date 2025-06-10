import pandas as pd
import os
import ast

def load_protein_data(data_path, level='subfamily'):
    """
    Loads protein data from data_new.csv, performs initial preparation,
    and handles basic preprocessing.

    Args:
        data_path (str): The path to the data_new.csv file.
        level (str): The classification level, 'subfamily' or 'family'.

    Returns:
        pd.DataFrame: The loaded and preprocessed DataFrame.
    """
    df = pd.read_csv(data_path)
    
    # Convert string representations of lists to actual lists
    df['Domains'] = df['Domains'].apply(ast.literal_eval)
    df['Seperators'] = df['Seperators'].apply(ast.literal_eval)
    
    # Create 'Family' column if classification is at the family level
    if level == 'family':
        df['Family'] = df['Subfamily'].apply(lambda x: '.'.join(x.split('.')[:3]))
        
    return df

def load_superfamily_map(map_path):
    """
    Loads the family to superfamily mapping from the CSV file.

    Args:
        map_path (str): The path to the fam2supefamily.csv file.

    Returns:
        dict: A dictionary mapping family IDs to superfamily labels.
    """
    try:
        superfamily_df = pd.read_csv(map_path)
        # Clean up whitespace in column names and values
        superfamily_df.columns = [col.strip() for col in superfamily_df.columns]
        family_to_superfamily = dict(zip(superfamily_df['family'].str.strip(), superfamily_df['label'].str.strip()))
        return family_to_superfamily
    except FileNotFoundError:
        print(f"Warning: Superfamily map file not found at {map_path}. Returning empty map.")
        return {}
    except Exception as e:
        print(f"Warning: Could not load superfamily data: {e}. Returning empty map.")
        return {} 