#!/usr/bin/env python3
import os
import shutil
import logging
import tensorflow as tf
from mediapipe_model_maker import gesture_recognizer

# Ensure TensorFlow 2.x
assert tf.__version__.startswith("2")

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

class GestureRecognitionTraining:
    def __init__(self, dataset_path, export_dir, task_filename,
                 epochs=100, batch_size=32, split_ratio=0.8):
        self.dataset_path = dataset_path
        self.export_dir = export_dir
        self.task_filename = task_filename
        self.epochs = epochs
        self.batch_size = batch_size
        self.split_ratio = split_ratio
        self.model = None
        self.train_data = None
        self.validation_data = None
        self.test_data = None

    def list_labels(self):
        labels = [
            d for d in os.listdir(self.dataset_path)
            if os.path.isdir(os.path.join(self.dataset_path, d))
        ]
        logger.info(f"Found {len(labels)} labels: {labels}")
        return labels

    def load_dataset(self):
        logger.info("Loading dataset...")
        data = gesture_recognizer.Dataset.from_folder(
            dirname=self.dataset_path,
            hparams=gesture_recognizer.HandDataPreprocessingParams()
        )
        train_data, rest_data = data.split(self.split_ratio)
        validation_data, test_data = rest_data.split(0.5)

        self.train_data = train_data
        self.validation_data = validation_data
        self.test_data = test_data
        logger.info("Dataset loaded and split into train/validation/test")

    def create_and_train_model(self):
        logger.info("Creating and training model...")
        hparams = gesture_recognizer.HParams(
            export_dir=self.export_dir,
            epochs=self.epochs,
            shuffle=True,
            batch_size=self.batch_size,
        )
        options = gesture_recognizer.GestureRecognizerOptions(hparams=hparams)
        self.model = gesture_recognizer.GestureRecognizer.create(
            train_data=self.train_data,
            validation_data=self.validation_data,
            options=options
        )
        logger.info("Model training complete")

    def evaluate_model(self):
        if not self.model:
            raise RuntimeError("Model is not trained yet.")
        logger.info("Evaluating model on test data...")
        loss, acc = self.model.evaluate(self.test_data, batch_size=1)
        logger.info(f"Test loss: {loss:.4f}, Test accuracy: {acc:.4f}")

    def export_model(self):
        if not self.model:
            raise RuntimeError("Model is not trained yet.")
        logger.info("Exporting model...")
        self.model.export_model()

        logger.info(
            f"Files in {self.export_dir}: {os.listdir(self.export_dir)}"
            )

        src_path = os.path.join(self.export_dir, "gesture_recognizer.task")
        dst_path = self.task_filename
        shutil.copy(src_path, dst_path)
        logger.info(f"Copied {src_path} → {dst_path}")

    def run(self):
        logger.info("Starting gesture recognizer pipeline...")
        self.list_labels()
        self.load_dataset()
        self.create_and_train_model()
        self.evaluate_model()
        self.export_model()
        logger.info("Pipeline complete.")

# -------------------------------
# MAIN
# -------------------------------
def main():
    pipeline = GestureRecognitionTraining(
        dataset_path="/home/megrad/Downloads/asl_letters_only",
        export_dir="exported_model",
        task_filename="asl_letters_only.task",
        epochs=100,
        batch_size=32,
    )
    pipeline.run()

if __name__ == "__main__":
    main()
