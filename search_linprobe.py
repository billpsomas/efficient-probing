import optuna
from optuna.samplers import GridSampler

import argparse
import os
from main_linprobe import get_args_parser, main
import torch.distributed as dist
import torch

# Parse command-line arguments as if running the original script
original_parser = get_args_parser()
original_args = original_parser.parse_args()  # Captures user-specified args

def objective(trial):
    # blr = trial.suggest_loguniform('blr', 1e-2, 0.1)  # Adjust range as needed
    # blr = trial.suggest_categorical('blr', [1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1])
    # blr = trial.suggest_float('blr', low=5.0, high=50.0)
    blr = trial.suggest_categorical('blr', search_space['blr'])

    # Create argument parser
    # args = get_args_parser().parse_args([])  # Pass empty list to avoid CLI conflicts
    args = argparse.Namespace(**vars(original_args))

    args.blr = blr
    args.lr = None  # Ensure it's recomputed based on `blr`

    args.output_dir = os.path.join(args.output_dir, f"trial_{trial.number}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(os.environ['CUDA_VISIBLE_DEVICES'])
    if dist.is_initialized():
        dist.destroy_process_group()    
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    # # Run training
    main(args)

    # Load validation accuracy from log file
    log_file_path = os.path.join(args.output_dir, "training_log.txt")
    with open(log_file_path, "r") as f:
        lines = f.readlines()
    
    # Extract last validation accuracy from log
    val_acc = 0.0
    for line in lines:
        if "Max Accuracy" in line:
            val_acc = float(line.split(":")[-1].strip().replace("%", ""))
            break

    return val_acc  # Optuna tries to maximize this value

if __name__ == "__main__":
    # SGD search space
    #search_space = {'blr': [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5]}
    #search_space = {'blr': [0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05]}
    # LARS search spaces
    #search_space = {'blr': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0]}
    search_space = {'blr': [0.1, 0.2, 0.5, 0.7, 1.0, 1.2, 1.5]}
    # search_space = {'blr': [0.05, 0.06, 0.07, 0.08, 0.09, 0.1]}
    # search_space = {'blr': [1.0, 1.1, 1.2, 1.3, 1.4]}
    study = optuna.create_study(direction="maximize", sampler=GridSampler(search_space))  # Maximize validation accuracy
    study.optimize(objective, n_trials=len(search_space['blr'])) 
    
    # study = optuna.create_study(direction="maximize")  # Maximize validation accuracy 
    # study.optimize(objective, n_trials=10)  # Adjust the number of trials as needed

    print("Best BLR:", study.best_params['blr'])
    print("Best Validation Accuracy:", study.best_value)