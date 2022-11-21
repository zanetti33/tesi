import argparse
import random
import torch
from torch import optim
from transformers import ViltConfig, ViltProcessor, ViltFeatureExtractor, AutoTokenizer
from data_collator import ViltDataCollatorForMetricLearning
from pytorch_metric_learning import losses, miners
import pytorch_metric_learning.utils.logging_presets as logging_presets
import wandb
from dataset import MimicCxrMetricLearningDataset
from metric_learning_trainer import (
    TwoStreamMetricLearningTester,
    TwoStreamMetricLearningTrainer,
)
from model import ViltModelForEmbedding
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
import os


def main():

    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    vilt_processor = ViltProcessor(
        ViltFeatureExtractor(resample=3, image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5], size_divisor=32),
        AutoTokenizer.from_pretrained("bert-base-uncased", model_max_length=args.max_position_embeddings),
    )
    # config modello
    config = ViltConfig(max_position_embeddings=args.max_position_embeddings)
    model = ViltModelForEmbedding.from_pretrained(args.pretrained_model, config=config)
    model.train()
    model.to(device)

    training_dataset = MimicCxrMetricLearningDataset(args.dataset_path, split="train")
    validation_dataset = MimicCxrMetricLearningDataset(args.dataset_path, split="validate")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
    )

    # Loss Function
    mining_func = miners.MultiSimilarityMiner()
    loss_func = losses.MultiSimilarityLoss(alpha=args.multsim_alpha, beta=args.multsim_beta, base=args.multsim_lambda)

    models = {"trunk": model}
    optimizers = {"trunk_optimizer": optimizer}
    loss_funcs = {"metric_loss": loss_func}
    mining_funcs = {"tuple_miner": mining_func}

    # Logging
    experiments_folder = os.path.join("metric_learning_experiments", args.experiment_name)
    tensorboard_folder = os.path.join(experiments_folder, "tensorboard")
    wandb_config = config.to_dict()
    wandb_config.update(args.__dict__)
    # Attach wandb to tensorboard (pytorch-metric-learning suppports only the latter)
    wandb.tensorboard.patch(root_logdir=tensorboard_folder)

    wandb.init(
        project=args.wandb_project_name,
        entity=args.wandb_entity,
        name=args.experiment_name,
        config=wandb_config,
        # sync_tensorboard=True,
    )
    record_keeper, _, tensorboard_writer = logging_presets.get_record_keeper(
        csv_folder=os.path.join(experiments_folder, "logs"),
        tensorboard_folder=tensorboard_folder,
        experiment_name=args.experiment_name,
    )
    tensorboard_writer.add_text("config", model.config.to_json_string(), 0)
    tensorboard_writer.add_text("flags", str(args.__dict__), 0)
    hooks = logging_presets.get_hook_container(record_keeper)

    # Create the tester
    tester = TwoStreamMetricLearningTester(
        end_of_testing_hook=hooks.end_of_testing_hook,
        normalize_embeddings=False,
        batch_size=args.eval_batch_size,
        data_device=device,
        use_trunk_output=True,
        dataloader_num_workers=args.dataloader_workers,
        accuracy_calculator=AccuracyCalculator(k=100, avg_of_avgs=True, device=device),
    )

    model_folder = os.path.join(experiments_folder, "best_models")
    end_of_epoch_hook = hooks.end_of_epoch_hook(
        tester,
        {"val": validation_dataset},
        model_folder,
        test_interval=1,
        patience=args.patience if args.patience > 0 else None,
        test_collate_fn=ViltDataCollatorForMetricLearning(vilt_processor),
    )
    trainer = TwoStreamMetricLearningTrainer(
        models,
        optimizers,
        args.train_batch_size,
        loss_funcs,
        mining_funcs,
        training_dataset,
        data_device=device,
        dataloader_num_workers=args.dataloader_workers,
        end_of_iteration_hook=hooks.end_of_iteration_hook,
        # end_of_iteration_hook=end_of_epoch_hook,
        end_of_epoch_hook=end_of_epoch_hook,
        collate_fn=ViltDataCollatorForMetricLearning(vilt_processor),
    )

    start_epoch = hooks.load_latest_saved_models(trainer, model_folder, device) if args.resume_from_checkpoint else 1
    trainer.train(start_epoch=start_epoch, num_epochs=args.num_epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Metric Learning")
    parser.add_argument(
        "--logging",
        default="online",
        type=str,
        choices=["disabled", "offline", "online"],
        help="Set 'disabled' to disable wandb logging, or else select logging 'online' or 'offline'",
    )
    parser.add_argument(
        "--wandb_project_name", default="radiography_retrieval", type=str, help="Project name for wandb logs"
    )
    parser.add_argument("--wandb_entity", default="tesi-zanetti", type=str, help="Wandb entity")
    parser.add_argument(
        "-en",
        "--experiment_name",
        help="defines the directory where the training checkpoint and logs are saved",
    )
    parser.add_argument(
        "--random_seed",
        default=42,
        type=int,
        help="Random seed",
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/datasets/MIMIC-CXR",
        help="Tokenizer used to tokenize texts",
    )
    parser.add_argument(
        "--vilt_pretrained_processor",
        type=str,
        default="dandelin/vilt-b32-mlm",
        help="ViLT preprocessor for texts and images",
    )
    parser.add_argument(
        "--max_position_embeddings",
        default=512,
        type=int,
        help="Maximum number of position embeddings",
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default="dandelin/vilt-b32-mlm",
        help="Path to a pretrained model",
    )
    parser.add_argument("-lr", "--learning_rate", type=float, default=2e-5, help="defines the learning rate") #5e-6
    parser.add_argument("-wd", "--weight_decay", type=float, default=1e-4, help="defines the weight decay")
    parser.add_argument("-b1", "--adam_beta1", type=float, default=0.9, help="defines the hyperparameter beta 1")
    parser.add_argument("-b2", "--adam_beta2", type=float, default=0.999, help="defines the hyperparameter beta 2")
    parser.add_argument("--multsim_alpha", default=2.0, type=float, help="Multisimilarity Alpha")
    parser.add_argument("--multsim_beta", default=40.0, type=float, help="Multisimilarity Beta")
    parser.add_argument("--multsim_lambda", default=0.5, type=float, help="Multisimilarity Lambda")
    parser.add_argument(
        "--num_epochs",
        default=10,
        type=int,
        help="Number of epochs",
    )
    parser.add_argument(
        "--patience",
        default=3,
        type=int,
        help="Patience",
    )
    parser.add_argument(
        "--dataloader_workers",
        default=1,
        type=int,
        help="Number of workers to use inside the dataloader",
    )
    parser.add_argument(
        "--eval_batch_size",
        default=32,
        type=int,
        help="Batch size used during validation",
    )
    parser.add_argument(
        "--train_batch_size",
        default=24,
        type=int,
        help="Batch size used during training",
    )
    parser.add_argument("--resume_from_checkpoint", action=argparse.BooleanOptionalAction)

    args = parser.parse_args()
    main()
