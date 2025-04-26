import os
import sys
import torch
import torchaudio
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.dataio.batch import PaddedData
import json
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import sklearn


class ParkinsonsBrain(sb.Brain):
    """
    A specialized SpeechBrain model for Parkinson's disease detection from speech audio.
    
    This class extends SpeechBrain's Brain class to implement custom forward pass,
    loss calculation, metric tracking, and demographic analysis for a Parkinson's 
    disease classification model. It includes tracking and visualization of 
    performance metrics across different demographic groups.
    
    Attributes:
        epoch_metrics (dict): Dictionary tracking loss and error metrics per epoch
        demographic_results (dict): Dictionary tracking performance by demographic groups
        plots_dir (str): Directory where performance plots are saved
        
    Args:
        modules (dict): The PyTorch modules that make up the model
        opt_class (torch.optim): The PyTorch optimizer class
        hparams (dict): Hyperparameters for training and evaluation
        run_opts (dict): Runtime options for SpeechBrain execution
        checkpointer (sb.utils.checkpoints): Checkpoint manager
        
    Example:
        >>> # Setup hyperparameters and modules
        >>> with open("hparams.yaml") as fin:
        ...     hparams = load_hyperpyyaml(fin)
        >>> # Prepare datasets
        >>> datasets = dataio_prep(hparams)
        >>> # Initialize model
        >>> pd_brain = ParkinsonsBrain(
        ...     modules=hparams["modules"],
        ...     opt_class=hparams["opt_class"],
        ...     hparams=hparams,
        ...     run_opts=run_opts,
        ...     checkpointer=hparams["checkpointer"],
        ... )
        >>> # Train and evaluate
        >>> pd_brain.fit(
        ...     epoch_counter=pd_brain.hparams.epoch_counter,
        ...     train_set=datasets["train"],
        ...     valid_set=datasets["valid"],
        ... )
        >>> result = pd_brain.evaluate(test_set=datasets["test"])
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add tracking for metrics over epochs
        self.epoch_metrics = {
            "train_loss": [],
            "valid_loss": [],
            "valid_error": [],
            "epoch": []
        }

        # Store demographic results
        self.demographic_results = {
            "sex": {"epoch": [], "groups": [], "accuracy": []},
            "age": {"epoch": [], "groups": [], "accuracy": []},
            "dataset": {"epoch": [], "groups": [], "accuracy": []}
        }

        # Make sure output directory exists
        self.plots_dir = os.path.join(self.hparams.output_folder, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)

    def compute_forward(self, batch, stage):
        """
        Forward computations from input audio to classification output.
        
        This method processes audio through the feature extraction pipeline,
        embedding model, and classifier to generate predictions.
        
        Args:
            batch (PaddedBatch): The input batch containing audio signals and metadata.
            stage (sb.Stage): The current stage (TRAIN, VALID, or TEST).
            
        Returns:
            torch.Tensor: The model's predictions (logits) with shape [batch_size, num_classes].
            
        Example:
            >>> # Typically called internally during training/evaluation
            >>> batch = next(iter(train_loader))
            >>> predictions = pd_brain.compute_forward(batch, sb.Stage.TRAIN)
            >>> print(f"Prediction shape: {predictions.shape}")
            Prediction shape: torch.Size([8, 2])
        """
        batch = batch.to(self.device)
        sigs, lengths = batch.sig
        # 1) Features [B, n_mels, T]
        fbanks = self.modules.compute_features(sigs)
        fbanks = torch.log(fbanks + 1e-10)
        fbanks = fbanks.transpose(1, 2)
        fbanks = self.modules.mean_var_norm(fbanks, lengths)
        # 2) ECAPA expects [B, T, n_mels], drop lengths arg here
        embeddings = self.modules.embedding_model(fbanks)
        # 3) Classify
        raw_pred = self.modules.classifier(embeddings)
        # 4) Unwrap only if it's a PaddedData
        if isinstance(raw_pred, PaddedData):
            predictions = raw_pred.data
        else:
            predictions = raw_pred
        return predictions

    def on_stage_start(self, stage, epoch=None):
        """
        Prepare for a new training, validation, or test stage.
        
        This method initializes the metrics and trackers needed for the current stage.
        
        Args:
            stage (sb.Stage): The current stage (TRAIN, VALID, or TEST).
            epoch (int, optional): The current epoch number. Defaults to None.
            
        Returns:
            None: This method initializes internal trackers but doesn't return a value.
            
        Example:
            >>> # Typically called internally at the start of each stage
            >>> pd_brain.on_stage_start(sb.Stage.VALID, epoch=5)
            >>> print(f"Initialized metrics for {sb.Stage.VALID}")
            Initialized metrics for Stage.VALID
        """
        self.loss_metric = sb.utils.metric_stats.MetricStats(
            metric=sb.nnet.losses.bce_loss
        )
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()

            # Only initialize these if they don't exist yet
            if not hasattr(self, "all_ids"):
                self.all_ids = []
                self.all_preds = []
                self.all_targets = []
                print(f"Initialized tracking lists for {stage}")
            else:
                print(
                    f"Using existing tracking lists with {len(self.all_ids)} items for {stage}")

    def compute_objectives(self, predictions, batch, stage):
        """
        Compute the loss and other metrics for the current batch.
        
        This method calculates the loss for training and tracks prediction accuracy
        for validation and testing. It also maintains lists of all predictions for
        later demographic analysis.
        
        Args:
            predictions (torch.Tensor): The model's output predictions.
            batch (PaddedBatch): The input batch with signals and labels.
            stage (sb.Stage): The current stage (TRAIN, VALID, or TEST).
            
        Returns:
            torch.Tensor: The computed loss value for this batch.
            
        Example:
            >>> # Typically called internally during training/evaluation
            >>> batch = next(iter(valid_loader))
            >>> predictions = pd_brain.compute_forward(batch, sb.Stage.VALID)
            >>> loss = pd_brain.compute_objectives(predictions, batch, sb.Stage.VALID)
            >>> print(f"Validation loss: {loss.item():.4f}")
            Validation loss: 0.6842
        """
        # Unwrap labels
        raw_labels = batch.label_encoded
        if isinstance(raw_labels, PaddedData):
            labels = raw_labels.data
        else:
            labels = raw_labels

        predictions = predictions.squeeze(1)  # Shape becomes [batch_size, 2]
        labels = labels.squeeze(-1)  # Shape becomes [batch_size]
        loss = sb.nnet.losses.nll_loss(
            torch.log_softmax(
                predictions, dim=-1), labels)

        if stage != sb.Stage.TRAIN:
            if not hasattr(self, "error_metrics"):
                self.error_metrics = self.hparams.error_stats()

            # Initialize tracking lists if they don't exist yet
            if not hasattr(self, "all_ids"):
                self.all_ids = []
                self.all_preds = []
                self.all_targets = []

            # Get argmax predictions
            pred_indices = torch.argmax(predictions, dim=-1)

            # Store in error_metrics
            self.error_metrics.append(batch.id, pred_indices, labels)

            # Also store directly in our tracking lists
            if isinstance(batch.id, list):
                self.all_ids.extend(batch.id)
            else:
                self.all_ids.append(batch.id)

            # Handle batched predictions
            for i in range(len(pred_indices)):
                pred = pred_indices[i].item() if pred_indices[i].numel(
                ) == 1 else pred_indices[i][0].item()
                target = labels[i].item() if labels[i].numel(
                ) == 1 else labels[i][0].item()
                self.all_preds.append(pred)
                self.all_targets.append(target)

        return loss

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """
        Process results at the end of each stage.
        
        This method updates learning rates, logs statistics, saves checkpoints,
        performs demographic analysis, and generates plots based on the results
        from the completed stage.
        
        Args:
            stage (sb.Stage): The stage that has just completed (TRAIN, VALID, or TEST).
            stage_loss (float): The average loss for the completed stage.
            epoch (int, optional): The current epoch number. Defaults to None.
            
        Returns:
            None: This method performs end-of-stage processing but doesn't return a value.
            
        Example:
            >>> # Typically called internally at the end of each stage
            >>> pd_brain.on_stage_end(sb.Stage.VALID, 0.5423, epoch=5)
            >>> print("Stage completed, model checkpointed")
            Stage completed, model checkpointed
        """
        # TRAIN stage: just record the loss
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss

            # Store for plotting
            if epoch is not None:
                self.epoch_metrics["train_loss"].append(float(stage_loss))
                self.epoch_metrics["epoch"].append(epoch)

            return

        # Calculate error rate from our tracked predictions
        error_count = 0
        total_count = len(self.all_preds)

        for i in range(total_count):
            if self.all_preds[i] != self.all_targets[i]:
                error_count += 1

        error_rate = error_count / total_count if total_count > 0 else 0

        stats = {
            "loss": stage_loss,
            "error": error_rate,
        }

        if stage == sb.Stage.VALID:
            # Store metrics for plotting
            if epoch is not None:
                if epoch not in self.epoch_metrics["epoch"]:
                    self.epoch_metrics["epoch"].append(epoch)
                self.epoch_metrics["valid_loss"].append(float(stage_loss))
                self.epoch_metrics["valid_error"].append(error_rate)

            # Adjust LR, log, checkpoint
            old_lr, new_lr = self.hparams.lr_annealing(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch, "lr": old_lr},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )
            self.checkpointer.save_and_keep_only(
                meta=stats, min_keys=["error"])

            # Demographic analysis
            self.analyze_demographics("valid", epoch=epoch)

            # Plot losses after each epoch
            self.plot_losses()

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )
            self.analyze_demographics("test")

            # Final plots for test set
            self.plot_demographic_results()

    def plot_losses(self):
        """
        Plot the training and validation losses and error rates.
        
        This method creates and saves a figure showing the progression of
        training loss, validation loss, and validation error over epochs.
        
        Args:
            None: This method uses the metrics tracked in self.epoch_metrics.
            
        Returns:
            None: This method saves a plot file but doesn't return a value.
            
        Example:
            >>> pd_brain.plot_losses()
            >>> print(f"Loss plots saved to {pd_brain.plots_dir}/loss_curves.png")
            Loss plots saved to exp/parkinsons_detection/plots/loss_curves.png
        """
        if not self.epoch_metrics["epoch"]:
            return

        plt.figure(figsize=(10, 6))
        epochs = self.epoch_metrics["epoch"]

        # Plot training loss
        if self.epoch_metrics["train_loss"]:
            plt.plot(
                epochs[:len(self.epoch_metrics["train_loss"])],
                self.epoch_metrics["train_loss"],
                'b-',
                label='Training Loss'
            )

        # Plot validation loss
        if self.epoch_metrics["valid_loss"]:
            plt.plot(
                epochs[:len(self.epoch_metrics["valid_loss"])],
                self.epoch_metrics["valid_loss"],
                'r-',
                label='Validation Loss'
            )

        # Plot validation error
        if self.epoch_metrics["valid_error"]:
            plt.plot(
                epochs[:len(self.epoch_metrics["valid_error"])],
                self.epoch_metrics["valid_error"],
                'g-',
                label='Validation Error'
            )

        plt.title('Training and Validation Metrics')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)

        # Save the figure
        plt.savefig(os.path.join(self.plots_dir, 'loss_curves.png'))
        plt.close()

    def plot_demographic_results(self):
        """
        Plot the performance across different demographic groups.
        
        This method generates plots showing model accuracy for different
        demographic categories (sex, age, and dataset group).
        
        Args:
            None: This method uses data tracked in self.demographic_results.
            
        Returns:
            None: This method saves plot files but doesn't return a value.
            
        Example:
            >>> pd_brain.plot_demographic_results()
            >>> print("Demographic analysis plots created")
            Demographic analysis plots created
        """
        # Plot sex-based results
        if self.demographic_results["sex"]["groups"]:
            self._plot_demographic_category("sex", "Sex-based Performance")

        # Plot age-based results
        if self.demographic_results["age"]["groups"]:
            self._plot_demographic_category("age", "Age-based Performance")

        # Plot dataset group-based results
        if self.demographic_results["dataset"]["groups"]:
            self._plot_demographic_category(
                "dataset", "Dataset Group-based Performance")

    def _plot_demographic_category(self, category, title):
        """
        Plot the performance for a specific demographic category.
        
        This helper method generates a plot showing model accuracy over time
        for each group within a demographic category (sex, age, or dataset).
        
        Args:
            category (str): The demographic category to plot ('sex', 'age', or 'dataset').
            title (str): The title for the plot.
            
        Returns:
            None: This method saves a plot file but doesn't return a value.
            
        Example:
            >>> pd_brain._plot_demographic_category('sex', 'Sex-based Performance')
            >>> print(f"Plot saved to {pd_brain.plots_dir}/sex_performance.png")
            Plot saved to exp/parkinsons_detection/plots/sex_performance.png
        """
        plt.figure(figsize=(12, 8))
        data = self.demographic_results[category]

        # Create a list of unique groups
        unique_groups = sorted(set(data["groups"]))

        # Group by epoch
        epochs = sorted(set(data["epoch"]))

        # Create a dictionary to hold accuracy by epoch and group
        grouped_data = {group: [] for group in unique_groups}

        for epoch, group, acc in zip(
                data["epoch"], data["groups"], data["accuracy"]):
            if group in grouped_data:
                grouped_data[group].append((epoch, acc))

        # Plot line for each group
        for group, values in grouped_data.items():
            if values:
                epochs, accuracies = zip(*values)
                plt.plot(epochs, accuracies, 'o-', label=f'{group}')

        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        # Save the figure
        plt.savefig(
            os.path.join(
                self.plots_dir,
                f'{category}_performance.png'))
        plt.close()

    def analyze_demographics(self, stage_name, epoch=None):
        """
        Analyze model performance across different demographic groups.
        
        This method calculates accuracy for different demographic groups 
        (sex, age, dataset) and optionally creates snapshot plots for the current epoch.
        
        Args:
            stage_name (str): The current stage name ('train', 'valid', or 'test').
            epoch (int, optional): The current epoch number. Defaults to None.
            
        Returns:
            None: This method prints analysis results and saves plots but doesn't return a value.
            
        Example:
            >>> pd_brain.analyze_demographics('valid', epoch=10)
            
            Demographic analysis for valid:
            - Processing 120 samples
            M: 0.88 (72/82)
            F: 0.76 (29/38)
            20s: 0.92 (23/25)
            30s: 0.85 (17/20)
            ...
        """
        if not hasattr(self, "all_ids") or not self.all_ids:
            print(f"No evaluation data available for {stage_name}")
            return

        print(f"\nDemographic analysis for {stage_name}:")
        print(f"- Processing {len(self.all_ids)} samples")

        # Group metrics by demographics
        by_sex = {
            "M": {
                "correct": 0, "total": 0}, "F": {
                "correct": 0, "total": 0}}
        by_age = {}  # Will populate with age groups
        by_group = {}  # Will populate with dataset groups

        # Process each sample
        for i, sample_id in enumerate(self.all_ids):
            if i >= len(self.all_preds) or i >= len(self.all_targets):
                continue

            # Get prediction and target as scalar values
            pred = self.all_preds[i].item() if torch.is_tensor(
                self.all_preds[i]) else self.all_preds[i]
            target = self.all_targets[i].item() if torch.is_tensor(
                self.all_targets[i]) else self.all_targets[i]

            # Extract demographic information from ID
            filename = str(sample_id)

            # Extract sex (M/F) from filename
            sex = "Unknown"
            if "M" in filename and "F" not in filename:
                sex = "M"
            elif "F" in filename:
                sex = "F"

            # Extract age from filename
            age = 0
            for j in range(len(filename) - 2):
                if j + 2 < len(filename) and filename[j:j + 2].isdigit():
                    if filename[j + 2] == "M" or filename[j + 2] == "F":
                        age = int(filename[j:j + 2])
                        break

            # Determine group based on filename
            group = "HC" if target == 0 else "PD"
            if "Healthy Control" in filename:
                group = "HC"
            elif "Parkinson" in filename:
                group = "PD"

            # Age groups by decade
            age_group = f"{int(age / 10) * 10}s" if age > 0 else "Unknown"

            # Initialize counters if needed
            if age_group not in by_age:
                by_age[age_group] = {"correct": 0, "total": 0}
            if group not in by_group:
                by_group[group] = {"correct": 0, "total": 0}

            # Update counters
            by_sex[sex]["total"] += 1
            by_age[age_group]["total"] += 1
            by_group[group]["total"] += 1

            if pred == target:
                by_sex[sex]["correct"] += 1
                by_age[age_group]["correct"] += 1
                by_group[group]["correct"] += 1

        print(f"\nDemographic analysis for {stage_name}:")
        print(f"- Processing {len(self.all_ids)} samples")

        # By Sex
        for sex, counts in by_sex.items():
            accuracy = counts["correct"] / \
                counts["total"] if counts["total"] > 0 else 0
            print(
                f" {sex}: {accuracy:.2f} ({counts['correct']}/{counts['total']})")

        # By Age Group
        for age_group in sorted(by_age.keys()):
            counts = by_age[age_group]
            accuracy = counts["correct"] / \
                counts["total"] if counts["total"] > 0 else 0
            print(
                f" {age_group}: {accuracy:.2f} ({counts['correct']}/{counts['total']})")

        # By Dataset Group
        for group in sorted(by_group.keys()):
            counts = by_group[group]
            accuracy = counts["correct"] / \
                counts["total"] if counts["total"] > 0 else 0
            print(
                f" {group}: {accuracy:.2f} ({counts['correct']}/{counts['total']})")

        # Plot a snapshot of current demographic results
        if epoch is not None:
            # Create a bar chart for this epoch
            plt.figure(figsize=(15, 5))

            # Plot 1: By Sex
            plt.subplot(131)
            sex_groups = []
            sex_accs = []
            for sex, counts in by_sex.items():
                if counts["total"] > 0:
                    accuracy = counts["correct"] / counts["total"]
                    sex_groups.append(sex)
                    sex_accs.append(accuracy)

            plt.bar(sex_groups, sex_accs)
            plt.title(f'Accuracy by Sex (Epoch {epoch})')
            plt.ylim(0, 1)

            # Plot 2: By Age Group
            plt.subplot(132)
            age_groups = []
            age_accs = []
            for age_group in sorted(by_age.keys()):
                counts = by_age[age_group]
                if counts["total"] > 0:
                    accuracy = counts["correct"] / counts["total"]
                    age_groups.append(age_group)
                    age_accs.append(accuracy)

            plt.bar(age_groups, age_accs)
            plt.title(f'Accuracy by Age Group (Epoch {epoch})')
            plt.ylim(0, 1)

            # Plot 3: By Dataset Group
            plt.subplot(133)
            data_groups = []
            data_accs = []
            for group in sorted(by_group.keys()):
                counts = by_group[group]
                if counts["total"] > 0:
                    accuracy = counts["correct"] / counts["total"]
                    data_groups.append(group)
                    data_accs.append(accuracy)

            plt.bar(data_groups, data_accs)
            plt.title(f'Accuracy by Dataset Group (Epoch {epoch})')
            plt.ylim(0, 1)

            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    self.plots_dir,
                    f'demographic_epoch_{epoch}.png'))
            plt.close()


def dataio_prep(hparams):
    """
    Prepare the data for training and evaluation.
    
    This function creates data pipelines for loading and processing audio files,
    handles label encoding, and constructs SpeechBrain datasets for training,
    validation, and testing.
    
    Args:
        hparams (dict): The hyperparameters containing paths and processing options.
        
    Returns:
        dict: A dictionary containing SpeechBrain DynamicItemDataset objects for 
              'train', 'valid', and 'test' splits.
        
    Example:
        >>> # Load hyperparameters
        >>> with open("hparams.yaml") as fin:
        ...     hparams = load_hyperpyyaml(fin)
        >>> # Prepare datasets
        >>> datasets = dataio_prep(hparams)
        >>> print(f"Created datasets with {len(datasets['train'])} training samples")
        Created datasets with 300 training samples
    """
    # Create the label encoder with unknown label handling
    label_encoder = sb.dataio.encoder.CategoricalEncoder()

    # Initialize the encoder with known labels (using the proper method)
    # SpeechBrain uses update_from_iterable, not add_item
    label_encoder.update_from_iterable(["HC", "PD"])

    # Make sure the encoder knows how many labels to expect
    label_encoder.expect_len(2)

    @sb.utils.data_pipeline.takes("path")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(path):
        """
        Load and process the audio file at the given path.
        """
        # print(f"Original path: {path}")

        # Handle virtual paths with the Italian dataset prefix
        if "Italian_Parkinsons_Voice_and_Speech" in path or "italian_parkinson" in path:
            # Base directory for the dataset
            base_dir = "/home/k_ammade/Projects/COMP691_Project/pd_dataset"

            # Extract the relevant parts from the path
            if "22 Elderly Healthy Control" in path:
                group = "22 Elderly Healthy Control"
            elif "15 Young Healthy Control" in path:
                group = "15 Young Healthy Control"
            elif "28 People with Parkinson's disease" in path or "28 People with Parkinson" in path:
                group = "28 People with Parkinson's disease"
            else:
                print(f"Unknown group in path: {path}")
                return torch.zeros(16000)  # Return dummy signal

            # Extract speaker and filename
            parts = path.split('/')
            try:
                # Find the group in parts
                group_index = next(
                    i for i, part in enumerate(parts) if group in part)
                if group_index + 1 < len(parts):
                    speaker = parts[group_index + 1]
                else:
                    print(f"Could not find speaker in path: {path}")
                    return torch.zeros(16000)

                # Get filename (last part)
                filename = parts[-1]

                # Construct new path
                new_path = os.path.join(base_dir, group, speaker, filename)
                # print(f"Transformed path: {new_path}")
                path = new_path
            except (StopIteration, IndexError) as e:
                print(f"Error parsing path {path}: {e}")
                return torch.zeros(16000)

        # Check if file exists
        if not os.path.exists(path):
            print(f"WARNING: File does not exist: {path}")
            return torch.zeros(16000)

        # Load audio
        try:
            sig, fs = torchaudio.load(path)
            sig = torchaudio.functional.resample(
                sig, orig_freq=fs, new_freq=16000).squeeze(0)
            # print(f"Successfully loaded audio from {path}")
            return sig
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return torch.zeros(16000)

    @sb.utils.data_pipeline.takes("label")
    @sb.utils.data_pipeline.provides("label", "label_encoded")
    def label_pipeline(label):
        yield label
        label_encoded = label_encoder.encode_label_torch(label)
        yield label_encoded

    # Create the datasets
    datasets = {}
    # Set shuffling for training
    hparams["dataloader_options"]["shuffle"] = True

    for split in ["train", "valid", "test"]:
        datasets[split] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=hparams[f"{split}_annotation"],
            dynamic_items=[audio_pipeline, label_pipeline],
            output_keys=["id", "sig", "label_encoded"],
        )

    for split in ("train", "valid", "test"):
        with open(f"manifests/{split}.json") as f:
            manifest = json.load(f)
        counts = Counter(entry["label"] for entry in manifest.values())
        print(f"{split:5} →", counts)

    # Save/load label encoder
    lab_enc_file = os.path.join(hparams["output_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[datasets["train"]],
        output_key="label",
    )

    # Print the encoded labels for debugging
    print("\nLabel Encoding:")
    for label, idx in label_encoder.lab2ind.items():
        print(f"  {label} -> {idx}")

    return datasets


if __name__ == "__main__":
    # Parse command-line args
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    # Load hyperparameters
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)
    # Set up experiment
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )
    # Prepare data
    datasets = dataio_prep(hparams)
    # Initialize Brain and start training
    pd_brain = ParkinsonsBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    pd_brain.fit(
        epoch_counter=pd_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )
    # Your original code checks if result is a dict or float:
result = pd_brain.evaluate(
    test_set=datasets["test"],
    min_key="error",
    test_loader_kwargs=hparams["dataloader_options"],
)

# if evaluate returned a dict, extract its 'error', else assume it's the
# float itself
if isinstance(result, dict):
    test_error = result["error"]
else:
    test_error = result

# print it
print(f"\nSaved test error: {test_error}")
print("\nTraining and evaluation complete!")
print(f"Final test error rate: {test_error:.4f}")
print(f"Final test accuracy: {1 - test_error:.4f}")

# Delete or comment out these lines (they're causing the error):
# print(f"Saved test error: {result['error']}")
# print("\nTraining and evaluation complete!")
# print(f"Final test error rate: {result['error']:.4f}")
# print(f"Final test accuracy: {1 - result['error']:.4f}")

# Create final summary plots
print("Creating final summary plots...")
# Plot confusion matrix for test set
if hasattr(pd_brain, "all_preds") and hasattr(pd_brain, "all_targets"):
    plt.figure(figsize=(8, 6))
    # Create confusion matrix
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    # Convert predictions and targets to numpy arrays
    y_pred = np.array([p.item() if torch.is_tensor(
        p) else p for p in pd_brain.all_preds])
    y_true = np.array([t.item() if torch.is_tensor(
        t) else t for t in pd_brain.all_targets])
    # Create and plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "HC",
            "PD"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix (Test Set)')
    # Save the figure
    plt.savefig(os.path.join(pd_brain.plots_dir, 'confusion_matrix.png'))
    plt.close()

# Plot ROC curve if we have probabilities
try:
    from sklearn.metrics import roc_curve, auc
    # Create final performance summary
    print("\nFinal Test Performance Summary:")
    print(f"Error Rate: {test_error:.4f}")
    print(f"Accuracy: {1 - test_error:.4f}")
    print("\nPlots saved to:", pd_brain.plots_dir)
except BaseException:
    print("Could not generate ROC curve (requires raw probabilities)")
    print("Training and evaluation complete!")
