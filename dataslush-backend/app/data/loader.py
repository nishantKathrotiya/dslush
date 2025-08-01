import pandas as pd
import os
from app.utils.cleaning import (
    remove_extra_headers,
    fix_column_names,
    clean_html,
    normalize_list_column
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_csv(file_name):
    path = os.path.join(BASE_DIR, 'files', file_name)
    return pd.read_csv(path)

def load_all_data():
    print("Loading and cleaning datasets...")

    talent_df = fix_column_names(load_csv('talent_samples.csv'))
    skills_df = remove_extra_headers(load_csv('skills.csv'))
    software_df = remove_extra_headers(load_csv('software.csv'))
    job_types_df = remove_extra_headers(load_csv('job_types.csv'))
    content_df = remove_extra_headers(load_csv('content_verticals.csv'))
    styles_df = remove_extra_headers(load_csv('creative_styles.csv'))
    creators_df = load_csv('creators_hiring_optional.csv')

    # Clean and normalize columns
    talent_df['Bio'] = talent_df['Bio'].fillna("").str.lower()
    talent_df['Skills'] = talent_df['Skills'].apply(normalize_list_column)
    talent_df['Software'] = talent_df['Software'].apply(normalize_list_column)
    talent_df['Content Verticals'] = talent_df['Content Verticals'].apply(normalize_list_column)
    talent_df['Creative Styles'] = talent_df['Creative Styles'].apply(normalize_list_column)
    talent_df['Platforms'] = talent_df['Platforms'].apply(normalize_list_column)

    # Strip HTML from job descriptions
    creators_df['Description'] = creators_df['Description'].apply(clean_html)

    print("✅ Datasets loaded and cleaned.")
    return {
        "talents": talent_df,
        "skills": skills_df,
        "software": software_df,
        "job_types": job_types_df,
        "content": content_df,
        "styles": styles_df,
        "creators": creators_df
    }
