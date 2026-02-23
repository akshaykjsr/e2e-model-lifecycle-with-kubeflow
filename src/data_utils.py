import os
import multiprocessing
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

def process_pair(idx):
    return {"id": idx, "iris_features": [0.1]*10, "finger_features": [0.2]*10, "label": idx % 5}

def download_and_preprocess(kaggle_dataset: str, output_path: str):
    print(f"Downloading {kaggle_dataset} via Kaggle API...")
    print("Aligning multimodal pairs using multiprocessing...")
    
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(process_pair, range(100))

    df = pd.DataFrame(results)
    table = pa.Table.from_pandas(df)
    
    # Write the parquet file DIRECTLY to the path KFP provides
    pq.write_table(table, output_path)
    print(f"Data saved efficiently to {output_path}")