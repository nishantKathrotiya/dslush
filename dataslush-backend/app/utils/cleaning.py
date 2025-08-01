import pandas as pd
import re
from bs4 import BeautifulSoup

def remove_extra_headers(df: pd.DataFrame) -> pd.DataFrame:
    """
    For small single-column files (like skills, job types, etc.),
    remove rows that are repeated headers or blanks.
    """
    if df.shape[1] == 1:
        col = df.columns[0]
        df = df[df[col].notnull()]
        df = df[df[col].str.lower() != col.lower()]
        df = df.drop_duplicates()
        df[col] = df[col].str.strip()
    return df.reset_index(drop=True)

def fix_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename problematic columns in talent_samples.
    """
    renames = {
        '# of Views by Creators': 'Views By Creators',
        'First Name': 'First Name',
        'Last Name': 'Last Name',
        'Profile Description': 'Bio',
        'Job Types': 'Job Types',
        'Skills': 'Skills',
        'Software': 'Software',
        'Content Verticals': 'Content Verticals',
        'Creative Styles': 'Creative Styles',
        'Platforms': 'Platforms',
        'Past Creators': 'Past Creators',
        'Monthly Rate': 'Monthly Rate',
        'Hourly Rate': 'Hourly Rate'
    }
    df = df.rename(columns=renames)
    return df

def clean_html(text: str) -> str:
    """
    Strip HTML tags from job descriptions or bios.
    """
    if pd.isna(text):
        return ""
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(separator=" ").strip()

def normalize_list_column(text: str) -> list:
    """
    Convert comma-separated string into list of cleaned items.
    """
    if pd.isna(text):
        return []
    return [item.strip().lower() for item in text.split(',') if item.strip()]

