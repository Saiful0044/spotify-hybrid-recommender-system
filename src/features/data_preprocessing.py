import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import logging
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from category_encoders.count import CountEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from scipy.sparse import save_npz

# Configure logging
logger = logging.getLogger('data_cleaning')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)
formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# cols to transform
frequency_encode_cols = ['year']
ohe_cols = ['artist', "time_signature", "key"]
tfidf_col = 'tags'
standard_scale_cols = ["duration_ms", "loudness", "tempo"]
min_max_scale_cols = ["danceability", "energy", "speechiness",
                     "acousticness", "instrumentalness", "liveness", "valence"]

# load data
def load_data(data_path: Path) -> pd.DataFrame:
    try:
        data = pd.read_csv(data_path)
        logger.info("Data loaded successfully")
        return data
    except FileNotFoundError as e:
        logger.error(f"File not found: {data_path}")
        raise
    except pd.errors.EmptyDataError as e:
        logger.error(f"File is empty: {data_path}")
        raise

# train preprocessor
def train_preprocessor(preprocessor, data: pd.DataFrame) -> ColumnTransformer:
    processing_data = preprocessor.fit(data)
    logger.info("Preprocessor fitted successfully")
    return processing_data

# perform transformation
def perform_transformation(preprocessor, data: pd.DataFrame):
    transform_data = preprocessor.transform(data)
    logger.info("Data transformed successfully")
    return transform_data

# save data
def save_data(data, file_path: Path) -> None:
    try:
        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(data, (np.ndarray, pd.DataFrame)):
            if isinstance(data, np.ndarray):
                np.save(file_path.with_suffix('.npy'), data)
            else:
                data.to_csv(file_path.with_suffix('.csv'), index=False)
        else:  # sparse matrix
            save_npz(file_path.with_suffix('.npz'), data)
            
        logger.info(f'Data saved successfully to location')
    except Exception as e:
        logger.error(f"Failed to save data to {file_path}: {e}")
        raise

if __name__ == '__main__':
    # root path
    root_path = Path(__file__).parent.parent.parent
    # data path
    data_path = root_path/'data'/'cleaned'/'songs_cleaned_data.csv'
    data = load_data(data_path=data_path)
    
    # Ensure columns are in correct format
    data[frequency_encode_cols] = data[frequency_encode_cols].astype(str)
    data[ohe_cols] = data[ohe_cols].astype(str)
    
    # remove columns
    cols_to_remove = ['track_id', 'name','spotify_preview_url']
    data.drop(columns=cols_to_remove, inplace=True)
    
    # transform the data
    transform = ColumnTransformer(transformers=[
        ('frequency_encode', CountEncoder(normalize=True), frequency_encode_cols),
        ('ohe', OneHotEncoder(handle_unknown='ignore'), ohe_cols),
        ('tfidf', TfidfVectorizer(max_features=85), tfidf_col),
        ('standard_scale', StandardScaler(), standard_scale_cols),
        ('min_max_scale', MinMaxScaler(), min_max_scale_cols)
    ], remainder='passthrough')
    
    # fit the preprocessor
    trained_preprocessor = train_preprocessor(preprocessor=transform, data=data)
    
    # transform the data
    data_trans = perform_transformation(preprocessor=trained_preprocessor, data=data)
    
    # save the transformed data
    processed_dir = root_path/'data'/'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the transformed data (which is likely a sparse matrix)
    data_trans_path = processed_dir/'transformed_data.npz'
    save_data(data=data_trans, file_path=data_trans_path)

    # Save the preprocessor
    models_dir = root_path/'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    preprocessor_path = models_dir/'preprocessor.joblib'
    joblib.dump(trained_preprocessor, preprocessor_path)
    logger.info(f"Preprocessor saved to location")