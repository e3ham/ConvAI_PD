#!/usr/bin/env python3
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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

# Brain class for Parkinson detection using Whisper features from SpeechBrain


class ParkinsonBrain(sb.Brain):
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
        # Initialize tracking attributes
        self.train_losses = []
        self.valid_losses = []
        self.valid_errors = []
        self.test_error = None
        self.test_loss = None
        self.all_ids = []
        self.all_preds = []
        self.all_targets = []
        # Make sure output directory exists
        self.plots_dir = os.path.join(self.hparams.output_folder, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)

    def compute_forward(self, batch, stage):
        """Forward computations from the waveform to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig

        # Forward pass through the Whisper model
        # SpeechBrain's Whisper implementation with encoder_only=True returns
        # just encoder outputs
        feats = self.modules.whisper(wavs)

        # Apply pooling (average along time dimension)
        feats = self.modules.avg_pool(feats, wav_lens)

        # Flatten tensors for the classifier
        feats = feats.view(feats.shape[0], -1)

        # Forward pass through the classifier
        outputs = self.modules.classifier(feats)

        # Apply log softmax
        outputs = self.hparams.log_softmax(outputs)
        return outputs

    def compute_objectives(self, predictions, batch, stage):
        """Compute the loss (NLL) given predictions and targets."""
        # Unwrap labels
        if isinstance(batch.label_encoded, tuple):
            labels, *_ = batch.label_encoded  # For tuple format
        else:
            if isinstance(batch.label_encoded, PaddedData):
                labels = batch.label_encoded.data
            else:
                labels = batch.label_encoded

        # Make sure labels are correctly shaped for NLL loss
        if len(labels.shape) > 1:
            labels = labels.squeeze(-1)

        # Ensure labels has at least 1 dimension (not a scalar tensor)
        if len(labels.shape) == 0:
            labels = labels.unsqueeze(0)

        # Compute NLL loss
        loss = self.hparams.compute_cost(predictions, labels)

        # For VALID/TEST, handle error metrics manually instead of using
        # self.error_metrics.append
        if stage != sb.Stage.TRAIN:
            # Initialize tracking lists if they don't exist yet
            if not hasattr(self, "all_ids"):
                self.all_ids = []
                self.all_preds = []
                self.all_targets = []

            # Process batch IDs - ensure it's a list
            batch_ids = batch.id if isinstance(batch.id, list) else [batch.id]

            # Get predictions as class indices
            pred_indices = torch.argmax(predictions, dim=-1)

            # Store predictions and targets for analysis
            for i in range(len(pred_indices)):
                # Get ID (ensure we stay within range of batch_ids)
                id_idx = min(i, len(batch_ids) - 1)
                id = batch_ids[id_idx]

                # Extract scalar values
                pred_val = pred_indices[i].item() if torch.is_tensor(
                    pred_indices[i]) else pred_indices[i]

                # Handle cases where labels might have fewer items than
                # predictions
                if i < len(labels):
                    target_val = labels[i].item() if torch.is_tensor(
                        labels[i]) else labels[i]
                else:
                    target_val = labels[-1].item() if torch.is_tensor(labels[-1]
                                                                      ) else labels[-1]

                # Store for later analysis
                self.all_ids.append(id)
                self.all_preds.append(pred_val)
                self.all_targets.append(target_val)

        return loss

    def fit_batch(self, batch):
        """Step both the main optimizer and ssl_optimizer."""
        # 1) forward + loss
        preds = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(preds, batch, sb.Stage.TRAIN)
        # 2) backprop
        loss.backward()
        # 3) step both
        self.optimizer.step()
        if hasattr(self, "ssl_optimizer"):
            self.ssl_optimizer.step()
        # 4) zero grads
        self.optimizer.zero_grad()
        if hasattr(self, "ssl_optimizer"):
            self.ssl_optimizer.zero_grad()
        return loss

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Process results for each stage."""
        # TRAIN stage: just record the loss
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
            self.train_losses.append(stage_loss)

            # Store for plotting
            if epoch is not None:
                self.epoch_metrics["train_loss"].append(float(stage_loss))
                if epoch not in self.epoch_metrics["epoch"]:
                    self.epoch_metrics["epoch"].append(epoch)
            return

        # Calculate error rate manually from our tracked predictions
        error_count = 0
        total_count = len(self.all_preds)
        for i in range(total_count):
            if self.all_preds[i] != self.all_targets[i]:
                error_count += 1
        error_rate = error_count / total_count if total_count > 0 else 0

        # Print some stats
        print(
            f"\nStage: {stage}, Total samples: {total_count}, Errors: {error_count}")
        print(f"Error rate: {error_rate:.4f}, Accuracy: {1-error_rate:.4f}")

        # Stats for both VALID and TEST
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

            # Save for JSON output later
            self.valid_losses.append(float(stage_loss))
            self.valid_errors.append(float(error_rate))

            # Adjust LR, log, checkpoint
            old_lr, new_lr = self.hparams.lr_annealing(stats["error"])
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            # Update LR for SSL model if we're fine-tuning it
            if hasattr(self, "ssl_optimizer"):
                old_lr_ssl, new_lr_ssl = self.hparams.lr_annealing_ssl(
                    stats["error"])
                sb.nnet.schedulers.update_learning_rate(
                    self.ssl_optimizer, new_lr_ssl)

            # Log training progress
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch, "lr": old_lr},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )

            # Save model checkpoint
            self.checkpointer.save_and_keep_only(
                meta=stats, min_keys=["error"])

            # Demographic analysis
            self.analyze_demographics("valid", epoch=epoch)

            # Plot losses after each epoch
            self.plot_losses()

        elif stage == sb.Stage.TEST:
            self.test_error = stats["error"]  # Save for JSON output
            self.test_loss = stats["loss"]

            # Log test results
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )

            # Final analysis and plots
            self.analyze_demographics("test")
            self.plot_demographic_results()
            self.plot_confusion_matrix()

    def on_stage_start(self, stage, epoch=None):
        """Setup for each stage (TRAIN, VALID, TEST)."""
        # Loss metric for all stages
        self.loss_metric = sb.utils.metric_stats.MetricStats(
            metric=sb.nnet.losses.nll_loss
        )

        # For validation and test, also initialize error metrics
        if stage != sb.Stage.TRAIN:
            self.error_metrics = sb.utils.metric_stats.MetricStats(
                metric=sb.nnet.losses.classification_error,
                n_jobs=1
            )

            # Reset tracking lists at the beginning of each stage
            # This ensures we only track the current stage's predictions
            self.all_ids = []
            self.all_preds = []
            self.all_targets = []
            print(f"Reset tracking lists for {stage}")

    def init_optimizers(self):
        """Initialize optimizers for model and SSL model if needed."""
        # Regular model optimizer
        self.optimizer = self.hparams.opt_class(
            self.hparams.model.parameters())

        # SSL model optimizer if not freezing
        if not self.hparams.freeze_ssl:
            self.ssl_optimizer = self.hparams.ssl_opt_class(
                self.modules.whisper.parameters()
            )

            if self.checkpointer is not None:
                self.checkpointer.add_recoverable(
                    "ssl_optimizer", self.ssl_optimizer)

        # Add optimizer to checkpointer
        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("optimizer", self.optimizer)

    def plot_losses(self):
        """Plot the training and validation losses."""
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

    def plot_confusion_matrix(self):
        """Plot confusion matrix for the test set."""
        if not hasattr(self, "all_preds") or not self.all_preds:
            return

        plt.figure(figsize=(8, 6))

        # Convert predictions and targets to numpy arrays
        y_pred = np.array([p if not isinstance(p, torch.Tensor)
                          else p.item() for p in self.all_preds])
        y_true = np.array([t if not isinstance(t, torch.Tensor)
                          else t.item() for t in self.all_targets])

        # Create and plot confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=[
                "HC", "PD"])
        disp.plot(cmap=plt.cm.Blues)
        plt.title('Confusion Matrix (Test Set)')

        # Save the figure
        plt.savefig(os.path.join(self.plots_dir, 'confusion_matrix.png'))
        plt.close()

    def plot_demographic_results(self):
        """Plot the demographic analysis results."""
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
        """Plot a specific demographic category."""
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
        """Analyze model performance across different demographic groups."""
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
        by_group = {
            "HC": {
                "correct": 0, "total": 0}, "PD": {
                "correct": 0, "total": 0}}

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
                    if j + \
                            2 < len(filename) and (filename[j + 2] == "M" or filename[j + 2] == "F"):
                        age = int(filename[j:j + 2])
                        break

            # Determine group based on filename - improved detection
            group = "Unknown"
            if "HC" in filename or "Healthy Control" in filename or "Control" in filename:
                group = "HC"
            elif "PD" in filename or "Parkinson" in filename or "disease" in filename:
                group = "PD"

            # If still unknown, try to identify from the label target
            if group == "Unknown" and target is not None:
                if target == 0:  # Assuming 0 is HC based on the label encoding shown
                    group = "HC"
                elif target == 1:  # Assuming 1 is PD based on the label encoding shown
                    group = "PD"

            # Age groups by decade
            age_group = f"{int(age / 10) * 10}s" if age > 0 else "Unknown"

            # Initialize counters if needed
            if age_group not in by_age:
                by_age[age_group] = {"correct": 0, "total": 0}

            if group not in by_group:
                by_group[group] = {"correct": 0, "total": 0}

            # Update counters
            if sex in by_sex:
                by_sex[sex]["total"] += 1
                if pred == target:
                    by_sex[sex]["correct"] += 1

            by_age[age_group]["total"] += 1
            by_group[group]["total"] += 1

            if pred == target:
                by_age[age_group]["correct"] += 1
                by_group[group]["correct"] += 1

        # Process results for plotting
        for sex, counts in by_sex.items():
            accuracy = counts["correct"] / \
                counts["total"] if counts["total"] > 0 else 0
            print(
                f" {sex}: {accuracy:.2f} ({counts['correct']}/{counts['total']})")

            # Store for plotting if we have an epoch
            if epoch is not None:
                self.demographic_results["sex"]["epoch"].append(epoch)
                self.demographic_results["sex"]["groups"].append(sex)
                self.demographic_results["sex"]["accuracy"].append(accuracy)

        for age_group in sorted(by_age.keys()):
            counts = by_age[age_group]
            accuracy = counts["correct"] / \
                counts["total"] if counts["total"] > 0 else 0
            print(
                f" {age_group}: {accuracy:.2f} ({counts['correct']}/{counts['total']})")

            # Store for plotting if we have an epoch
            if epoch is not None:
                self.demographic_results["age"]["epoch"].append(epoch)
                self.demographic_results["age"]["groups"].append(age_group)
                self.demographic_results["age"]["accuracy"].append(accuracy)

        for group in sorted(by_group.keys()):
            counts = by_group[group]
            accuracy = counts["correct"] / \
                counts["total"] if counts["total"] > 0 else 0
            print(
                f" {group}: {accuracy:.2f} ({counts['correct']}/{counts['total']})")

            # Store for plotting if we have an epoch
            if epoch is not None:
                self.demographic_results["dataset"]["epoch"].append(epoch)
                self.demographic_results["dataset"]["groups"].append(group)
                self.demographic_results["dataset"]["accuracy"].append(
                    accuracy)

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
    """Prepare the data for training and evaluation."""
    # Create the label encoder
    label_encoder = sb.dataio.encoder.CategoricalEncoder()

    # 1. Define the audio pipeline
    @sb.utils.data_pipeline.takes("path")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        """Load and process the audio file."""
        try:
            sig = sb.dataio.dataio.read_audio(wav)

            # Resample if necessary
            if hparams.get("sample_rate", 16000) != 16000:
                sig = torchaudio.functional.resample(
                    sig.unsqueeze(0),
                    orig_freq=16000,
                    new_freq=hparams["sample_rate"]
                ).squeeze(0)

            return sig
        except Exception as e:
            print(f"Error loading {wav}: {e}")
            return torch.zeros(16000)  # Return dummy signal on error

    # 2. Define the label pipeline
    @sb.utils.data_pipeline.takes("label")
    @sb.utils.data_pipeline.provides("label", "label_encoded")
    def label_pipeline(label):
        yield label
        yield label_encoder.encode_label_torch(label)

    # 3. Create datasets for each split
    datasets = {}
    data_info = {
        "train": hparams["train_annotation"],
        "valid": hparams["valid_annotation"],
        "test": hparams["test_annotation"],
    }

    # Set shuffling for training
    hparams["dataloader_options"]["shuffle"] = True

    for dataset in data_info:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=data_info[dataset],
            replacements={"data_root": hparams.get("data_folder", "")},
            dynamic_items=[audio_pipeline, label_pipeline],
            output_keys=["id", "sig", "label_encoded"],
        )

    # Print dataset statistics
    for split in ("train", "valid", "test"):
        try:
            with open(data_info[split]) as f:
                manifest = json.load(f)
                counts = Counter(entry["label"] for entry in manifest.values())
                print(f"{split:5} →", counts)
        except Exception as e:
            print(f"Could not load statistics for {split}: {e}")

    # Save/load label encoder
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[datasets["train"]],
        output_key="label",
    )

    # Print the encoded labels for debugging
    print("\nLabel Encoding:")
    for label, idx in label_encoder.lab2ind.items():
        print(f" {label} -> {idx}")

    return datasets


if __name__ == "__main__":
    # Parse command-line args
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Load hyperparameters
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Prepare data
    datasets = dataio_prep(hparams)

    # Initialize Brain and start training
    model = ParkinsonBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Initialize optimizers
    model.init_optimizers()

    # Train model
    model.fit(
        epoch_counter=model.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"]
    )

    # Evaluate on test set
    test_stats = model.evaluate(
        test_set=datasets["test"],
        min_key="error",
        test_loader_kwargs=hparams["dataloader_options"],
    )

    # Save metrics to JSON files
    train_losses_path = os.path.join(
        hparams["output_folder"],
        "train_losses.json")
    valid_losses_path = os.path.join(
        hparams["output_folder"],
        "valid_losses.json")
    valid_errors_path = os.path.join(
        hparams["output_folder"],
        "valid_errors.json")
    test_error_path = os.path.join(hparams["output_folder"], "test_error.json")

    # Try to save training losses, with error handling
    try:
        with open(train_losses_path, "w") as f:
            json.dump([float(loss) for loss in model.train_losses], f)
        print(f"Saved {len(model.train_losses)} training losses")
    except Exception as e:
        print(f"Error saving train_losses: {e}")

    try:
        with open(valid_losses_path, "w") as f:
            json.dump([float(loss) for loss in model.valid_losses], f)
        print(f"Saved {len(model.valid_losses)} validation losses")
    except Exception as e:
        print(f"Error saving valid_losses: {e}")

    try:
        with open(valid_errors_path, "w") as f:
            json.dump([float(error) for error in model.valid_errors], f)
        print(f"Saved {len(model.valid_errors)} validation errors")
    except Exception as e:
        print(f"Error saving valid_errors: {e}")

    try:
        with open(test_error_path, "w") as f:
            json.dump(float(model.test_error)
                      if model.test_error is not None else None, f)
        print(f"Saved test error: {model.test_error}")
    except Exception as e:
        print(f"Error saving test_error: {e}")

    print("\nTraining and evaluation complete!")
    print(f"Final test error rate: {model.test_error:.4f}")
    print(f"Final test accuracy: {1 - model.test_error:.4f}")
    print(f"Plots saved to: {model.plots_dir}")

# #!/usr/bin/env python3
# import os
# import sys
# import torch
# import torchaudio
# import speechbrain as sb
# from hyperpyyaml import load_hyperpyyaml
# from speechbrain.dataio.batch import PaddedData
# import json
# from collections import Counter
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from pathlib import Path
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

# # Import our acoustic feature extractor
# from whisper_acoustic_features.acoustic_features import AcousticFeatureExtractor

# # Brain class for Parkinson detection using Whisper + acoustic features
# class ParkinsonBrain(sb.Brain):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         # Add tracking for metrics over epochs
#         self.epoch_metrics = {
#             "train_loss": [],
#             "valid_loss": [],
#             "valid_error": [],
#             "epoch": []
#         }
#         # Store demographic results
#         self.demographic_results = {
#             "sex": {"epoch": [], "groups": [], "accuracy": []},
#             "age": {"epoch": [], "groups": [], "accuracy": []},
#             "dataset": {"epoch": [], "groups": [], "accuracy": []}
#         }
#         # Initialize tracking attributes
#         self.train_losses = []
#         self.valid_losses = []
#         self.valid_errors = []
#         self.test_error = None
#         self.test_loss = None
#         self.all_ids = []
#         self.all_preds = []
#         self.all_targets = []
#         # Make sure output directory exists
#         self.plots_dir = os.path.join(self.hparams.output_folder, "plots")
#         os.makedirs(self.plots_dir, exist_ok=True)

#     def compute_forward(self, batch, stage):
#         """Forward computations from the waveform to the output probabilities."""
#         batch = batch.to(self.device)
#         wavs, wav_lens = batch.sig

#         # 1) Get Whisper embeddings
#         feats = self.modules.whisper(wavs)

#         # 2) Apply pooling to Whisper features
#         pooled = self.modules.avg_pool(feats, wav_lens)
#         pooled = pooled.view(pooled.shape[0], -1)  # (B, D_whisper)

#         # 3) Extract acoustic features
#         ac_feats = self.modules.ac_feats(wavs, wav_lens)
#         # Flatten if necessary
#         if len(ac_feats.shape) > 2:
#             ac_feats = ac_feats.view(ac_feats.shape[0], -1)

#         # 4) Concatenate features
#         fused = torch.cat([pooled, ac_feats], dim=1)

#         # 5) Classify
#         outputs = self.modules.classifier(fused)

#         # Apply log softmax
#         outputs = self.hparams.log_softmax(outputs)
#         return outputs

#     def compute_objectives(self, predictions, batch, stage):
#         """Compute the loss (NLL) given predictions and targets."""
#         # Unwrap labels
#         if isinstance(batch.label_encoded, tuple):
#             labels, *_ = batch.label_encoded  # For tuple format
#         else:
#             if isinstance(batch.label_encoded, PaddedData):
#                 labels = batch.label_encoded.data
#             else:
#                 labels = batch.label_encoded

#         # Make sure labels are correctly shaped for NLL loss
#         if len(labels.shape) > 1:
#             labels = labels.squeeze(-1)

#         # Ensure labels has at least 1 dimension (not a scalar tensor)
#         if len(labels.shape) == 0:
#             labels = labels.unsqueeze(0)

#         # Compute NLL loss
#         loss = self.hparams.compute_cost(predictions, labels)

#         # For VALID/TEST, handle error metrics manually instead of using self.error_metrics.append
#         if stage != sb.Stage.TRAIN:
#             # Initialize tracking lists if they don't exist yet
#             if not hasattr(self, "all_ids"):
#                 self.all_ids = []
#                 self.all_preds = []
#                 self.all_targets = []

#             # Process batch IDs - ensure it's a list
#             batch_ids = batch.id if isinstance(batch.id, list) else [batch.id]

#             # Get predictions as class indices
#             pred_indices = torch.argmax(predictions, dim=-1)

#             # Store predictions and targets for analysis
#             for i in range(len(pred_indices)):
#                 # Get ID (ensure we stay within range of batch_ids)
#                 id_idx = min(i, len(batch_ids) - 1)
#                 id = batch_ids[id_idx]

#                 # Extract scalar values
#                 pred_val = pred_indices[i].item() if torch.is_tensor(pred_indices[i]) else pred_indices[i]

#                 # Handle cases where labels might have fewer items than predictions
#                 if i < len(labels):
#                     target_val = labels[i].item() if torch.is_tensor(labels[i]) else labels[i]
#                 else:
#                     target_val = labels[-1].item() if torch.is_tensor(labels[-1]) else labels[-1]

#                 # Store for later analysis
#                 self.all_ids.append(id)
#                 self.all_preds.append(pred_val)
#                 self.all_targets.append(target_val)

#         return loss

#     def fit_batch(self, batch):
#         """Step both the main optimizer and ssl_optimizer."""
#         # 1) forward + loss
#         preds = self.compute_forward(batch, sb.Stage.TRAIN)
#         loss = self.compute_objectives(preds, batch, sb.Stage.TRAIN)
#         # 2) backprop
#         loss.backward()
#         # 3) step both
#         self.optimizer.step()
#         if hasattr(self, "ssl_optimizer"):
#             self.ssl_optimizer.step()
#         # 4) zero grads
#         self.optimizer.zero_grad()
#         if hasattr(self, "ssl_optimizer"):
#             self.ssl_optimizer.zero_grad()
#         return loss

#     def on_stage_end(self, stage, stage_loss, epoch=None):
#         """Process results for each stage."""
#         # TRAIN stage: just record the loss
#         if stage == sb.Stage.TRAIN:
#             self.train_loss = stage_loss
#             self.train_losses.append(stage_loss)

#             # Store for plotting
#             if epoch is not None:
#                 self.epoch_metrics["train_loss"].append(float(stage_loss))
#                 if epoch not in self.epoch_metrics["epoch"]:
#                     self.epoch_metrics["epoch"].append(epoch)
#             return

#         # Calculate error rate manually from our tracked predictions
#         error_count = 0
#         total_count = len(self.all_preds)
#         for i in range(total_count):
#             if self.all_preds[i] != self.all_targets[i]:
#                 error_count += 1
#         error_rate = error_count / total_count if total_count > 0 else 0

#         # Print some stats
#         print(f"\nStage: {stage}, Total samples: {total_count}, Errors: {error_count}")
#         print(f"Error rate: {error_rate:.4f}, Accuracy: {1-error_rate:.4f}")

#         # Stats for both VALID and TEST
#         stats = {
#             "loss": stage_loss,
#             "error": error_rate,
#         }

#         if stage == sb.Stage.VALID:
#             # Store metrics for plotting
#             if epoch is not None:
#                 if epoch not in self.epoch_metrics["epoch"]:
#                     self.epoch_metrics["epoch"].append(epoch)
#                 self.epoch_metrics["valid_loss"].append(float(stage_loss))
#                 self.epoch_metrics["valid_error"].append(error_rate)

#             # Save for JSON output later
#             self.valid_losses.append(float(stage_loss))
#             self.valid_errors.append(float(error_rate))

#             # Adjust LR, log, checkpoint
#             old_lr, new_lr = self.hparams.lr_annealing(stats["error"])
#             sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

#             # Update LR for SSL model if we're fine-tuning it
#             if hasattr(self, "ssl_optimizer"):
#                 old_lr_ssl, new_lr_ssl = self.hparams.lr_annealing_ssl(stats["error"])
#                 sb.nnet.schedulers.update_learning_rate(self.ssl_optimizer, new_lr_ssl)

#             # Log training progress
#             self.hparams.train_logger.log_stats(
#                 {"Epoch": epoch, "lr": old_lr},
#                 train_stats={"loss": self.train_loss},
#                 valid_stats=stats,
#             )

#             # Save model checkpoint
#             self.checkpointer.save_and_keep_only(meta=stats, min_keys=["error"])

#             # Demographic analysis
#             self.analyze_demographics("valid", epoch=epoch)

#             # Plot losses after each epoch
#             self.plot_losses()

#         elif stage == sb.Stage.TEST:
#             self.test_error = stats["error"]  # Save for JSON output
#             self.test_loss = stats["loss"]

#             # Log test results
#             self.hparams.train_logger.log_stats(
#                 {"Epoch loaded": self.hparams.epoch_counter.current},
#                 test_stats=stats,
#             )

#             # Final analysis and plots
#             self.analyze_demographics("test")
#             self.plot_demographic_results()
#             self.plot_confusion_matrix()

#     def on_stage_start(self, stage, epoch=None):
#         """Setup for each stage (TRAIN, VALID, TEST)."""
#         # Loss metric for all stages
#         self.loss_metric = sb.utils.metric_stats.MetricStats(
#             metric=sb.nnet.losses.nll_loss
#         )

#         # For validation and test, also initialize error metrics
#         if stage != sb.Stage.TRAIN:
#             self.error_metrics = sb.utils.metric_stats.MetricStats(
#                 metric=sb.nnet.losses.classification_error,
#                 n_jobs=1
#             )

#             # Reset tracking lists at the beginning of each stage
#             # This ensures we only track the current stage's predictions
#             self.all_ids = []
#             self.all_preds = []
#             self.all_targets = []
#             print(f"Reset tracking lists for {stage}")

#     def init_optimizers(self):
#         """Initialize optimizers for model and SSL model if needed."""
#         # Regular model optimizer
#         self.optimizer = self.hparams.opt_class(self.hparams.model.parameters())

#         # SSL model optimizer if not freezing
#         if not self.hparams.freeze_ssl:
#             self.ssl_optimizer = self.hparams.ssl_opt_class(
#                 self.modules.whisper.parameters()
#             )

#             if self.checkpointer is not None:
#                 self.checkpointer.add_recoverable("ssl_optimizer", self.ssl_optimizer)

#         # Add optimizer to checkpointer
#         if self.checkpointer is not None:
#             self.checkpointer.add_recoverable("optimizer", self.optimizer)

#     def plot_losses(self):
#         """Plot the training and validation losses."""
#         if not self.epoch_metrics["epoch"]:
#             return

#         plt.figure(figsize=(10, 6))
#         epochs = self.epoch_metrics["epoch"]

#         # Plot training loss
#         if self.epoch_metrics["train_loss"]:
#             plt.plot(
#                 epochs[:len(self.epoch_metrics["train_loss"])],
#                 self.epoch_metrics["train_loss"],
#                 'b-',
#                 label='Training Loss'
#             )

#         # Plot validation loss
#         if self.epoch_metrics["valid_loss"]:
#             plt.plot(
#                 epochs[:len(self.epoch_metrics["valid_loss"])],
#                 self.epoch_metrics["valid_loss"],
#                 'r-',
#                 label='Validation Loss'
#             )

#         # Plot validation error
#         if self.epoch_metrics["valid_error"]:
#             plt.plot(
#                 epochs[:len(self.epoch_metrics["valid_error"])],
#                 self.epoch_metrics["valid_error"],
#                 'g-',
#                 label='Validation Error'
#             )

#         plt.title('Training and Validation Metrics')
#         plt.xlabel('Epoch')
#         plt.ylabel('Value')
#         plt.legend()
#         plt.grid(True)

#         # Save the figure
#         plt.savefig(os.path.join(self.plots_dir, 'loss_curves.png'))
#         plt.close()

#     def plot_confusion_matrix(self):
#         """Plot confusion matrix for the test set."""
#         if not hasattr(self, "all_preds") or not self.all_preds:
#             return

#         plt.figure(figsize=(8, 6))

#         # Convert predictions and targets to numpy arrays
#         y_pred = np.array([p if not isinstance(p, torch.Tensor) else p.item() for p in self.all_preds])
#         y_true = np.array([t if not isinstance(t, torch.Tensor) else t.item() for t in self.all_targets])

#         # Create and plot confusion matrix
#         cm = confusion_matrix(y_true, y_pred)
#         disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["HC", "PD"])
#         disp.plot(cmap=plt.cm.Blues)
#         plt.title('Confusion Matrix (Test Set)')

#         # Save the figure
#         plt.savefig(os.path.join(self.plots_dir, 'confusion_matrix.png'))
#         plt.close()

#     def plot_demographic_results(self):
#         """Plot the demographic analysis results."""
#         # Plot sex-based results
#         if self.demographic_results["sex"]["groups"]:
#             self._plot_demographic_category("sex", "Sex-based Performance")

#         # Plot age-based results
#         if self.demographic_results["age"]["groups"]:
#             self._plot_demographic_category("age", "Age-based Performance")

#         # Plot dataset group-based results
#         if self.demographic_results["dataset"]["groups"]:
#             self._plot_demographic_category("dataset", "Dataset Group-based Performance")

#     def _plot_demographic_category(self, category, title):
#         """Plot a specific demographic category."""
#         plt.figure(figsize=(12, 8))
#         data = self.demographic_results[category]

#         # Create a list of unique groups
#         unique_groups = sorted(set(data["groups"]))

#         # Group by epoch
#         epochs = sorted(set(data["epoch"]))

#         # Create a dictionary to hold accuracy by epoch and group
#         grouped_data = {group: [] for group in unique_groups}

#         for epoch, group, acc in zip(data["epoch"], data["groups"], data["accuracy"]):
#             if group in grouped_data:
#                 grouped_data[group].append((epoch, acc))

#         # Plot line for each group
#         for group, values in grouped_data.items():
#             if values:
#                 epochs, accuracies = zip(*values)
#                 plt.plot(epochs, accuracies, 'o-', label=f'{group}')

#         plt.title(title)
#         plt.xlabel('Epoch')
#         plt.ylabel('Accuracy')
#         plt.legend()
#         plt.grid(True)

#         # Save the figure
#         plt.savefig(os.path.join(self.plots_dir, f'{category}_performance.png'))
#         plt.close()

#     def analyze_demographics(self, stage_name, epoch=None):
#         """Analyze model performance across different demographic groups."""
#         if not hasattr(self, "all_ids") or not self.all_ids:
#             print(f"No evaluation data available for {stage_name}")
#             return

#         print(f"\nDemographic analysis for {stage_name}:")
#         print(f"- Processing {len(self.all_ids)} samples")

#         # Group metrics by demographics
#         by_sex = {"M": {"correct": 0, "total": 0}, "F": {"correct": 0, "total": 0}}
#         by_age = {}  # Will populate with age groups
#         by_group = {"HC": {"correct": 0, "total": 0}, "PD": {"correct": 0, "total": 0}}

#         # Process each sample
#         for i, sample_id in enumerate(self.all_ids):
#             if i >= len(self.all_preds) or i >= len(self.all_targets):
#                 continue

#             # Get prediction and target as scalar values
#             pred = self.all_preds[i].item() if torch.is_tensor(self.all_preds[i]) else self.all_preds[i]
#             target = self.all_targets[i].item() if torch.is_tensor(self.all_targets[i]) else self.all_targets[i]

#             # Extract demographic information from ID
#             filename = str(sample_id)

#             # Extract sex (M/F) from filename
#             sex = "Unknown"
#             if "M" in filename and "F" not in filename:
#                 sex = "M"
#             elif "F" in filename:
#                 sex = "F"

#             # Extract age from filename
#             age = 0
#             for j in range(len(filename) - 2):
#                 if j+2 < len(filename) and filename[j:j+2].isdigit():
#                     if j+2 < len(filename) and (filename[j+2] == "M" or filename[j+2] == "F"):
#                         age = int(filename[j:j+2])
#                         break

#             # Determine group based on filename - improved detection
#             group = "Unknown"
#             if "HC" in filename or "Healthy Control" in filename or "Control" in filename:
#                 group = "HC"
#             elif "PD" in filename or "Parkinson" in filename or "disease" in filename:
#                 group = "PD"

#             # If still unknown, try to identify from the label target
#             if group == "Unknown" and target is not None:
#                 if target == 0:  # Assuming 0 is HC based on the label encoding shown
#                     group = "HC"
#                 elif target == 1:  # Assuming 1 is PD based on the label encoding shown
#                     group = "PD"

#             # Age groups by decade
#             age_group = f"{int(age / 10) * 10}s" if age > 0 else "Unknown"

#             # Initialize counters if needed
#             if age_group not in by_age:
#                 by_age[age_group] = {"correct": 0, "total": 0}

#             if group not in by_group:
#                 by_group[group] = {"correct": 0, "total": 0}

#             # Update counters
#             if sex in by_sex:
#                 by_sex[sex]["total"] += 1
#                 if pred == target:
#                     by_sex[sex]["correct"] += 1

#             by_age[age_group]["total"] += 1
#             by_group[group]["total"] += 1

#             if pred == target:
#                 by_age[age_group]["correct"] += 1
#                 by_group[group]["correct"] += 1

#         # Process results for plotting
#         for sex, counts in by_sex.items():
#             accuracy = counts["correct"] / counts["total"] if counts["total"] > 0 else 0
#             print(f" {sex}: {accuracy:.2f} ({counts['correct']}/{counts['total']})")

#             # Store for plotting if we have an epoch
#             if epoch is not None:
#                 self.demographic_results["sex"]["epoch"].append(epoch)
#                 self.demographic_results["sex"]["groups"].append(sex)
#                 self.demographic_results["sex"]["accuracy"].append(accuracy)

#         for age_group in sorted(by_age.keys()):
#             counts = by_age[age_group]
#             accuracy = counts["correct"] / counts["total"] if counts["total"] > 0 else 0
#             print(f" {age_group}: {accuracy:.2f} ({counts['correct']}/{counts['total']})")

#             # Store for plotting if we have an epoch
#             if epoch is not None:
#                 self.demographic_results["age"]["epoch"].append(epoch)
#                 self.demographic_results["age"]["groups"].append(age_group)
#                 self.demographic_results["age"]["accuracy"].append(accuracy)

#         for group in sorted(by_group.keys()):
#             counts = by_group[group]
#             accuracy = counts["correct"] / counts["total"] if counts["total"] > 0 else 0
#             print(f" {group}: {accuracy:.2f} ({counts['correct']}/{counts['total']})")

#             # Store for plotting if we have an epoch
#             if epoch is not None:
#                 self.demographic_results["dataset"]["epoch"].append(epoch)
#                 self.demographic_results["dataset"]["groups"].append(group)
#                 self.demographic_results["dataset"]["accuracy"].append(accuracy)

#         # Plot a snapshot of current demographic results
#         if epoch is not None:
#             # Create a bar chart for this epoch
#             plt.figure(figsize=(15, 5))

#             # Plot 1: By Sex
#             plt.subplot(131)
#             sex_groups = []
#             sex_accs = []
#             for sex, counts in by_sex.items():
#                 if counts["total"] > 0:
#                     accuracy = counts["correct"] / counts["total"]
#                     sex_groups.append(sex)
#                     sex_accs.append(accuracy)
#             plt.bar(sex_groups, sex_accs)
#             plt.title(f'Accuracy by Sex (Epoch {epoch})')
#             plt.ylim(0, 1)

#             # Plot 2: By Age Group
#             plt.subplot(132)
#             age_groups = []
#             age_accs = []
#             for age_group in sorted(by_age.keys()):
#                 counts = by_age[age_group]
#                 if counts["total"] > 0:
#                     accuracy = counts["correct"] / counts["total"]
#                     age_groups.append(age_group)
#                     age_accs.append(accuracy)
#             plt.bar(age_groups, age_accs)
#             plt.title(f'Accuracy by Age Group (Epoch {epoch})')
#             plt.ylim(0, 1)

#             # Plot 3: By Dataset Group
#             plt.subplot(133)
#             data_groups = []
#             data_accs = []
#             for group in sorted(by_group.keys()):
#                 counts = by_group[group]
#                 if counts["total"] > 0:
#                     accuracy = counts["correct"] / counts["total"]
#                     data_groups.append(group)
#                     data_accs.append(accuracy)
#             plt.bar(data_groups, data_accs)
#             plt.title(f'Accuracy by Dataset Group (Epoch {epoch})')
#             plt.ylim(0, 1)

#             plt.tight_layout()
#             plt.savefig(os.path.join(self.plots_dir, f'demographic_epoch_{epoch}.png'))
#             plt.close()


# def dataio_prep(hparams):
#     """Prepare the data for training and evaluation."""
#     # Create the label encoder
#     label_encoder = sb.dataio.encoder.CategoricalEncoder()

#     # 1. Define the audio pipeline
#     @sb.utils.data_pipeline.takes("path")
#     @sb.utils.data_pipeline.provides("sig")
#     def audio_pipeline(wav):
#         """Load and process the audio file."""
#         try:
#             sig = sb.dataio.dataio.read_audio(wav)

#             # Resample if necessary
#             if hparams.get("sample_rate", 16000) != 16000:
#                 sig = torchaudio.functional.resample(
#                     sig.unsqueeze(0),
#                     orig_freq=16000,
#                     new_freq=hparams["sample_rate"]
#                 ).squeeze(0)

#             return sig
#         except Exception as e:
#             print(f"Error loading {wav}: {e}")
#             return torch.zeros(16000)  # Return dummy signal on error

#     # 2. Define the label pipeline
#     @sb.utils.data_pipeline.takes("label")
#     @sb.utils.data_pipeline.provides("label", "label_encoded")
#     def label_pipeline(label):
#         yield label
#         yield label_encoder.encode_label_torch(label)

#     # 3. Create datasets for each split
#     datasets = {}
#     data_info = {
#         "train": hparams["train_annotation"],
#         "valid": hparams["valid_annotation"],
#         "test": hparams["test_annotation"],
#     }

#     # Set shuffling for training
#     hparams["dataloader_options"]["shuffle"] = True

#     for dataset in data_info:
#         datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
#             json_path=data_info[dataset],
#             replacements={"data_root": hparams.get("data_folder", "")},
#             dynamic_items=[audio_pipeline, label_pipeline],
#             output_keys=["id", "sig", "label_encoded"],
#         )

#     # Print dataset statistics
#     for split in ("train","valid","test"):
#         try:
#             with open(data_info[split]) as f:
#                 manifest = json.load(f)
#                 counts = Counter(entry["label"] for entry in manifest.values())
#                 print(f"{split:5} →", counts)
#         except Exception as e:
#             print(f"Could not load statistics for {split}: {e}")

#     # Save/load label encoder
#     lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
#     label_encoder.load_or_create(
#         path=lab_enc_file,
#         from_didatasets=[datasets["train"]],
#         output_key="label",
#     )

#     # Print the encoded labels for debugging
#     print("\nLabel Encoding:")
#     for label, idx in label_encoder.lab2ind.items():
#         print(f" {label} -> {idx}")

#     return datasets


# if __name__ == "__main__":
#     # Parse command-line args
#     hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

#     # Load hyperparameters
#     with open(hparams_file) as fin:
#         hparams = load_hyperpyyaml(fin, overrides)

#     # Create experiment directory
#     sb.create_experiment_directory(
#         experiment_directory=hparams["output_folder"],
#         hyperparams_to_save=hparams_file,
#         overrides=overrides,
#     )

#     # Prepare data
#     datasets = dataio_prep(hparams)

#     # Initialize Brain and start training
#     model = ParkinsonBrain(
#         modules=hparams["modules"],
#         opt_class=hparams["opt_class"],
#         hparams=hparams,
#         run_opts=run_opts,
#         checkpointer=hparams["checkpointer"],
#     )

#     # Initialize optimizers
#     model.init_optimizers()

#     # Train model
#     model.fit(
#         epoch_counter=model.hparams.epoch_counter,
#         train_set=datasets["train"],
#         valid_set=datasets["valid"],
#         train_loader_kwargs=hparams["dataloader_options"],
#         valid_loader_kwargs=hparams["dataloader_options"]
#     )

#     # Evaluate on test set
#     test_stats = model.evaluate(
#         test_set=datasets["test"],
#         min_key="error",
#         test_loader_kwargs=hparams["dataloader_options"],
#     )

#     # Save metrics to JSON files
#     train_losses_path = os.path.join(hparams["output_folder"], "train_losses.json")
#     valid_losses_path = os.path.join(hparams["output_folder"], "valid_losses.json")
#     valid_errors_path = os.path.join(hparams["output_folder"], "valid_errors.json")
#     test_error_path = os.path.join(hparams["output_folder"], "test_error.json")

#     # Try to save training losses, with error handling
#     try:
#         with open(train_losses_path, "w") as f:
#             json.dump([float(loss) for loss in model.train_losses], f)
#         print(f"Saved {len(model.train_losses)} training losses")
#     except Exception as e:
#         print(f"Error saving train_losses: {e}")

#     try:
#         with open(valid_losses_path, "w") as f:
#             json.dump([float(loss) for loss in model.valid_losses], f)
#         print(f"Saved {len(model.valid_losses)} validation losses")
#     except Exception as e:
#         print(f"Error saving valid_losses: {e}")

#     try:
#         with open(valid_errors_path, "w") as f:
#             json.dump([float(error) for error in model.valid_errors], f)
#         print(f"Saved {len(model.valid_errors)} validation errors")
#     except Exception as e:
#         print(f"Error saving valid_errors: {e}")

#     try:
#         with open(test_error_path, "w") as f:
#             json.dump(float(model.test_error) if model.test_error is not None else None, f)
#         print(f"Saved test error: {model.test_error}")
#     except Exception as e:
#         print(f"Error saving test_error: {e}")

#     print("\nTraining and evaluation complete!")
#     print(f"Final test error rate: {model.test_error:.4f}")
#     print(f"Final test accuracy: {1 - model.test_error:.4f}")
#     print(f"Plots saved to: {model.plots_dir}")
