from train import evaluate_model

if __name__ == "__main__":
    base_params = {
        "input_size": 64,
        "num_filters": 16,
        "block_size": 1,
        "lstm_layers": 2,
        "lstm_hidden_size": 64,
        "num_outputs": 163,
    }
    share_params = {
        # "share_layers": [False, True],
        # "share_layer_batch_norm": [False, True],
        "share_layer_stride": 1,
        "share_rnn_stride": 2,
        "share_layer_batch_norm": False,
        "use_weight_scaler": False,
    }
    train_params = {
        # "batch_size": [32],
        "batch_size": 64,
        "optimizer": "adam",
        "epochs": 1000,
        "min_epochs": 80,
        "patience": 20,
        "optimizer_params": {
                "lr": 0.0003,
        },
        "scheduler_params": {"milestones": [1000000]},
        "augment": "resnet_simple",
        "model_id": 0,
    }

    model_name = "crnn"
    # dataset_names = ["IAM", "WASHINGTON", "PARZIVAL", "SAINT_GALL"]
    dataset_name = "IAM_S"
    experiment_name = "CRNN - IAM_S"

    evaluate_model(
        model_name=model_name,
        dataset_name=dataset_name,
        base_params=base_params,
        share_params=share_params,
        train_params=train_params,
        mlflow_experiment=experiment_name,
    )