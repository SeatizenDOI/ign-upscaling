import time
from pathlib import Path
from huggingface_hub import HFSummaryWriter

from .trainer import setup_trainer
from .dataset import DatasetManager
from .hugging_model_manager import ModelManager
from ..ConfigParser import ConfigParser
from ..utils.lib_tools import print_gpu_is_used
from .model_card_generator import generate_model_card, save_hyperparameters_to_config
from .evaluate import save_evaluation_to_text
from ..utils.training_step import TrainingStep


def main_launch_training(cp: ConfigParser, path_to_image: Path, training_step: TrainingStep) -> Path:
    """
        cp: From 
        path_to_image: Path to a folder with a subfolder images and a subfolder annotations
    """
    print("\n\n------ [TRAIN] ------\n\n")

    start_time = time.time()
    print_gpu_is_used()

    print("\n\n------ [TRAIN - Setup image dataset] ------\n")

    dataset_manager = DatasetManager(cp, path_to_image)
    dataset_manager.load_datasets()
    dataset_manager.attach_transforms()

    print("\n\n------ [TRAIN - Setup model] ------\n")

    model_manager = ModelManager(cp, training_step)
    model_manager.setup_model_dir()
    model_manager.setup_model(dataset_manager.num_labels)

    # Load Huggingface token.
    if model_manager.push_to_hub():
        logger = HFSummaryWriter(
            repo_id=model_manager.get_model_name_with_username(), 
            logdir=str(Path(model_manager.output_dir, "runs")), 
            commit_every=1
        )
    
    print("\n\n------ [TRAIN - Setup trainer] ------\n")

    trainer = setup_trainer(cp, dataset_manager, model_manager)

    print("\n\n------ [TRAIN - Start training] ------\n")

    if model_manager.latest_checkpoint != None:
        print("\ninfo : Resuming training from checkpoint \n", model_manager.latest_checkpoint)
        train_results = trainer.train(resume_from_checkpoint=model_manager.resume_from_checkpoint)
    else :
        train_results = trainer.train()

    print(f"Total training time: {time.time() - start_time} seconds")


    print("\n\n------ [TRAIN - Saving model] ------\n")
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()
    trainer.save_model(model_manager.output_dir)


    # For the first training, we don't care about generate model card, instead return model name.
    if training_step == TrainingStep.COARSE:
    
        trainer.evaluate()

        # === Predict on test set ===
        print(f"\n📊 Evaluating test set")
        predictions = trainer.predict(dataset_manager.validation_ds)
        save_evaluation_to_text(predictions, dataset_manager.num_labels, Path(model_manager.output_dir, 'results.txt'), model_manager.model_name)

        return model_manager.output_dir


    # Save hyperparameters.
    save_hyperparameters_to_config(model_manager.output_dir,cp)

    # Generate model card.
    files = ['train_results.json', 'trainer_state.json', 'all_results.json', 'config.json', 'transforms.json']
    data_paths = [Path(model_manager.output_dir, file) for file in files]
    
    print("\n\n------ [TRAIN - Generating model card] ------\n")
    generate_model_card(data_paths, model_manager)

    # Send data to hugging face if needed.
    if not model_manager.push_to_hub(): return model_manager.output_dir 
    print("\n\n------ [TRAIN - Send data to huggingface] ------\n")
    model_manager.send_data_to_hugging_face()

    return model_manager.output_dir