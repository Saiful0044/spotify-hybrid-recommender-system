import numpy as np
import pandas as pd
from pathlib import Path
import logging


# Configure logging
logger = logging.getLogger('data_cleaning')
logger.setLevel(logging.INFO)

# console handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

# add handler to logger
logger.addHandler(handler)

# create a fomratter
formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

def load_data(file_path: Path) -> pd.DataFrame:
    try:
        logger.info(f"Data Load Successfully")
        return pd.read_csv(file_path)
    except FileNotFoundError as e:
        logger.error(f"File not found: {file_path}")
        raise
    except pd.errors.EmptyDataError as e:
        logger.error(f"Empty file: {file_path}")
        raise


# data cleaning
def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    return(
        data
        .drop_duplicates(subset='track_id')
        .drop(columns=['genre','spotify_id'])
        .fillna({'tags': 'no_tags'})
        .assign(
            name = lambda x: x['name'].str.lower(),
            artist = lambda x: x['artist'].str.lower(),
            tags = lambda x: x['tags'].str.lower()
        )
        .reset_index(drop=True)
    )

# save data
def save_data(data: pd.DataFrame, file_path: Path)->None:
    try:
        data.to_csv(file_path, index=False)
        logger.info('Data saved successfully to location')

    except Exception as e:
        logger.error(f"Failed to save data to {file_path}: {e}")
        raise
    


# main code
if __name__=="__main__":
    # root path 
    root_path = Path(__file__).parent.parent.parent
    # songs data path
    songs_data_path = root_path/'data'/'raw'/'songs_data.csv'
    songs_data = load_data(file_path=songs_data_path)


    # data save directory
    data_save_dir = root_path/'data'/'cleaned'
    data_save_dir.mkdir(exist_ok=True,parents=True)
    cleaned_data_filename = 'songs_cleaned_data.csv'
    save_data_path = data_save_dir/cleaned_data_filename
    save_data(data=songs_data, file_path=save_data_path)




    

