from kfp import dsl
from kfp.dsl import Output, Input, Dataset

# BUMP TO v3!
BASE_IMAGE = "biometrics-pipeline:v3" 

@dsl.component(base_image=BASE_IMAGE)
def preprocess_data_op(dataset_name: str, processed_data: Output[Dataset]):
    from src.data_utils import download_and_preprocess
    
    download_and_preprocess(
        kaggle_dataset=dataset_name,
        output_path=processed_data.path  # Pass the artifact path
    )

@dsl.component(base_image=BASE_IMAGE)
def train_model_op(
    processed_data: Input[Dataset],
    epochs: int,
    batch_size: int
):
    from src.train import train_model
    
    train_model(
        data_path=processed_data.path, # Pass the artifact path
        epochs=epochs,
        batch_size=batch_size
    )