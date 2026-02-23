from kfp import dsl, compiler, Client
import components

@dsl.pipeline(
    name="Multimodal Biometrics Pipeline",
    description="An MLOps pipeline for multimodal data using PyArrow and PyTorch."
)
def biometrics_pipeline(
    dataset_name: str = "ninadmehendale/multimodal-iris-fingerprint-biometric-data",
    epochs: int = 5,
    batch_size: int = 32
):
    # Component 1: Download and Preprocess
    preprocess_task = components.preprocess_data_op(
        dataset_name=dataset_name
    )
    
    # Component 2: Train Model
    train_task = components.train_model_op(
        processed_data=preprocess_task.outputs["processed_data"], 
        epochs=epochs,
        batch_size=batch_size
    )

# Added to avoid OOM
    train_task.set_memory_request('2G')
    train_task.set_memory_limit('4G')
    train_task.set_cpu_request('1')
    train_task.set_cpu_limit('2')

if __name__ == "__main__":
    pipeline_filename = "biometrics_pipeline.yaml"
    compiler.Compiler().compile(
        pipeline_func=biometrics_pipeline,
        package_path=pipeline_filename
    )
    
    print(f"Compiled {pipeline_filename} successfully.")
    
    # connecting to local Kubeflow in this instance my local machine
    # used port forwarding for ingress/egress rules exception
    client = Client(host="http://localhost:8080") 
    
    print("Submitting pipeline run...")
    run_result = client.create_run_from_pipeline_func(
        biometrics_pipeline,
        arguments={"epochs": 10, "batch_size": 64},
        experiment_name="Biometrics_Local_Dev"
    )
    print(f"Run submitted! View details at: {run_result.run_id}")