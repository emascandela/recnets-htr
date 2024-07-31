def cfresnet():
    base_params = {
        "variant": ["r20", "r32", "r44"],
        "input_size": [64],
        "lstm_layers": [1, 2],
        "lstm_hidden_size": [64, 128, 256],
        "first_layer_stride": [2],
        "num_outputs": [163],
    }
    share_params = {
        # "share_layers": [False, True],
        # "share_layer_batch_norm": [False, True],
        "share_layer_stride": [1, 2, 4, 6],
        "share_rnn_stride": [1, 2],
        "share_layer_batch_norm": [False],
        "use_weight_scaler": [False, True],
    }
    train_params = {
        "batch_size": [32],
        "val_size": [0.1],
        "optimizer": ["adam"],
        "epochs": [20],
        "optimizer_params": [
            {
                "lr": 0.001,
            }
        ],
        "scheduler_params": [{"milestones": [5, 10, 18]}],
        "augment": ["resnet_simple"],
    }

    model_name = "cf_resnet"
    dataset_names = ["IAM", "PARZIVAL", "SAINT_GALL", "WASHINGTON"]
    experiment_name = "CFResNet"
    return (
        base_params,
        share_params,
        train_params,
        model_name,
        dataset_names,
        experiment_name,
    )


def crnn():
    base_params = {
        "input_size": [64],
        "num_filters": [32, 64, 128],
        "block_size": [1, 2, 3],
        "lstm_layers": [1, 2],
        "lstm_hidden_size": [64, 128, 256],
        "num_outputs": [163],
    }
    share_params = {
        # "share_layers": [False, True],
        # "share_layer_batch_norm": [False, True],
        "share_layer_stride": [1, 2, 3],
        "share_rnn_stride": [1, 2],
        "share_layer_batch_norm": [False],
        "use_weight_scaler": [False, True],
    }
    train_params = {
        "batch_size": [32],
        "val_size": [0.1],
        "optimizer": ["adam"],
        "epochs": [20],
        "optimizer_params": [
            {
                "lr": 0.001,
            }
        ],
        "scheduler_params": [{"milestones": [5, 10, 18]}],
        "augment": ["resnet_simple"],
    }

    model_name = "crnn"
    dataset_names = ["IAM", "PARZIVAL", "SAINT_GALL", "WASHINGTON"]
    experiment_name = "CRNN"
    return (
        base_params,
        share_params,
        train_params,
        model_name,
        dataset_names,
        experiment_name,
    )


def baseline():
    base_params = {
        "input_size": [64],
        "num_filters": [16],
        "block_size": reversed([1, 2, 3]),
        "lstm_layers": reversed([2, 3]),
        "lstm_hidden_size": [64, 256],
        "num_outputs": [163],
    }
    share_params = {
        # "share_layers": [False, True],
        # "share_layer_batch_norm": [False, True],
        "share_layer_stride": [1, 2, 3],
        "share_rnn_stride": [1, 2, 3],
        "share_layer_batch_norm": [False],
        "use_weight_scaler": [False, True],
    }
    train_params = {
        "batch_size": [32],
        # "batch_size": [64],
        "optimizer": ["adam"],
        "epochs": [1000],
        "min_epochs": [80],
        "patience": [20],
        "optimizer_params": [
            {
                "lr": 0.0003,
            }
        ],
        "scheduler_params": [{"milestones": [1000000]}],
        "augment": ["resnet_simple"],
        "model_id": [0],
    }

    model_name = "crnn"
    dataset_names = ["WASHINGTON", "PARZIVAL", "SAINT_GALL"]
    # dataset_names = ["IAM_S"]
    experiment_name = "CRNN"
    return (
        base_params,
        share_params,
        train_params,
        model_name,
        dataset_names,
        experiment_name,
    )


def find_best_model():
    base_params = {
        "input_size": [64],
        "num_filters": [64, 128],
        "const_filters": [True],
        # "num_filters": [[64, 64, 64, 64, 64]],
        "block_size": reversed([3, 2]),
        "lstm_layers": reversed([4, 3, 2]),
        # "lstm_hidden_size": [256],
        "lstm_hidden_size": [64, 128, 256],
        "num_outputs": [163],
    }
    share_params = {
        # "share_layers": [False, True],
        # "share_layer_batch_norm": [False, True],
        "share_layer_stride": [1],
        "share_rnn_stride": [1],
        "share_layer_batch_norm": [False],
        "use_weight_scaler": [False],


        # "cluster_ratio": [1.0, 0.25, 0.5, 0.1],
        # # "share_mode": reversed(["all"]),
        # "cnn_share_mode": ["all", "channel", "all"], #, "unit"], # !
        # "lstm_share_mode": ["all", "channel", "all"], #, "unit"], # !
        # "cluster_mode": ["all"], 
        # "cluster_steps": reversed([20, 50, 0]),
        # "warmup_steps": reversed([160, 0]), # !
        "cluster_ratio": [1.0],
        # "share_mode": reversed(["all"]),
        "cnn_share_mode": ["all"], #, "unit"], # !
        "lstm_share_mode": ["all"], #, "unit"], # !
        "share_cnn": [False],
        "share_lstm": [False],
        "cluster_mode": ["all"], 
        "cluster_steps": [0],
        "warmup_steps": [0],
    }
    train_params = {
        "batch_size": [32],
        # "batch_size": [64],
        "optimizer": ["adam"],
        "epochs": [1000],
        "min_epochs": [200],
        "patience": [50],
        "optimizer_params": [
            {
                "lr": 0.0003,
            }
        ],
        "scheduler_params": [{"milestones": [1000000]}],
        "augment": ["resnet_simple"],
        "fold": [0],
    }

    model_name = "crnn"
    dataset_names = ["WASHINGTON"]
    # dataset_names = ["IAM_S"]
    experiment_name = "CRNN"
    return (
        base_params,
        share_params,
        train_params,
        model_name,
        dataset_names,
        experiment_name,
    )

def wclus_reduction():
    base_params = {
        "input_size": [64],
        "num_filters": [128],
        "const_filters": [True],
        # "num_filters": [[64, 64, 64, 64, 64]],
        "block_size": [3],
        "lstm_layers": [2],
        # "lstm_hidden_size": [256],
        "lstm_hidden_size": [128],
        "num_outputs": [163],
    }
    share_params = {
        # "share_layers": [False, True],
        # "share_layer_batch_norm": [False, True],
        "share_layer_stride": [1],
        "share_rnn_stride": [1],
        "share_layer_batch_norm": [False],
        "use_weight_scaler": [False],


        # "cluster_ratio": [1.0, 0.25, 0.5, 0.1],
        # # "share_mode": reversed(["all"]),
        # "cnn_share_mode": ["all", "channel", "all"], #, "unit"], # !
        # "lstm_share_mode": ["all", "channel", "all"], #, "unit"], # !
        # "cluster_mode": ["all"], 
        # "cluster_steps": reversed([20, 50, 0]),
        # "warmup_steps": reversed([160, 0]), # !
        "cluster_ratio": [0.5, 0.75, 0.25, 0.1],
        # "share_mode": reversed(["all"]),
        "cnn_share_mode": ["all", "channel"],
        "lstm_share_mode": ["all"],
        ##### OPTIONAL
        # "lstm_share_mode": ["all", "channel"],
        ##### OPTIONAL
        "share_cnn": [False, True],
        "share_lstm": [False, True],
        "cluster_mode": ["all"], 
        "cluster_steps": [0],
        "warmup_steps": [250],

        # "cluster_steps": [20],
        # "warmup_steps": [0],
    }
    train_params = {
        "batch_size": [32],
        # "batch_size": [64],
        "optimizer": ["adam"],
        "epochs": [1000],
        # "epochs": [10],
        "min_epochs": [400],
        "patience": [50],
        "optimizer_params": [
            {
                "lr": 0.0003,
            }
        ],
        "scheduler_params": [{"milestones": [1000000]}],
        "augment": ["resnet_simple"],
        "fold": [0],
    }

    model_name = "crnn"
    dataset_names = ["WASHINGTON"]
    # dataset_names = ["IAM_S"]
    experiment_name = "CRNN"
    return (
        base_params,
        share_params,
        train_params,
        model_name,
        dataset_names,
        experiment_name,
    )

def baseline_reduction():
    base_params = {
        "input_size": [64],
        "num_filters": [128, 64, 32, 16],
        "const_filters": [True],
        # "num_filters": [[64, 64, 64, 64, 64]],
        "block_size": [3, 2, 1],
        "lstm_layers": [2, 1],
        # "lstm_hidden_size": [256],
        "lstm_hidden_size": [128, 64, 32, 16],
        "num_outputs": [163],
    }
    share_params = {
        "share_layer_stride": [1],
        "share_rnn_stride": [1],
        "share_layer_batch_norm": [False],
        "use_weight_scaler": [False],
        "cluster_ratio": [1.0],
        "cnn_share_mode": ["all"],
        "lstm_share_mode": ["all"],
        "share_cnn": [False],
        "share_lstm": [False],
        "cluster_mode": ["all"], 
        "cluster_steps": [0],
        "warmup_steps": [0],
    }
    train_params = {
        "batch_size": [32],
        # "batch_size": [64],
        "optimizer": ["adam"],
        "epochs": [1000],
        "min_epochs": [400],
        "patience": [50],
        "optimizer_params": [
            {
                "lr": 0.0003,
            }
        ],
        "scheduler_params": [{"milestones": [1000000]}],
        "augment": ["resnet_simple"],
        "fold": [0],
    }

    model_name = "crnn"
    dataset_names = ["WASHINGTON"]
    # dataset_names = ["IAM_S"]
    experiment_name = "CRNN"
    return (
        base_params,
        share_params,
        train_params,
        model_name,
        dataset_names,
        experiment_name,
    )


def baseline_recursion():
    base_params = {
        "input_size": [64],
        "num_filters": [128, 64, 32],
        "const_filters": [True],
        # "num_filters": [[64, 64, 64, 64, 64]],
        "block_size": [3, 2],
        "lstm_layers": [2],
        # "lstm_hidden_size": [256],
        "lstm_hidden_size": [64, 128],
        "num_outputs": [163],
    }
    share_params = {
        # "share_layers": [False, True],
        # "share_layer_batch_norm": [False, True],
        "share_layer_stride": [1, 2, 3],
        "share_rnn_stride": [1, 2],
        "share_layer_batch_norm": [False],
        "use_weight_scaler": [False, True],


        "cluster_ratio": [1.0],
        "cnn_share_mode": ["all"], #, "unit"], # !
        "lstm_share_mode": ["all"], #, "unit"], # !
        "share_cnn": [False],
        "share_lstm": [False],
        "cluster_mode": ["all"], 
        "cluster_steps": [0],
        "warmup_steps": [0],
    }
    train_params = {
        "batch_size": [32],
        # "batch_size": [64],
        "optimizer": ["adam"],
        "epochs": [1000],
        "min_epochs": [400],
        "patience": [50],
        "optimizer_params": [
            {
                "lr": 0.0003,
            }
        ],
        "scheduler_params": [{"milestones": [1000000]}],
        "augment": ["resnet_simple"],
        "fold": [0],
    }

    model_name = "crnn"
    dataset_names = ["WASHINGTON"]
    # dataset_names = ["IAM_S"]
    experiment_name = "CRNN"
    return (
        base_params,
        share_params,
        train_params,
        model_name,
        dataset_names,
        experiment_name,
    )
