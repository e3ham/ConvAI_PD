#!/usr/bin/env python3
import os
import sys
import subprocess
import re


def main():
    # Load the template YAML
    try:
        with open("ecapa.yaml", "r") as fin:
            content = fin.read()
    except FileNotFoundError:
        try:
            with open("paste.txt", "r") as fin:
                content = fin.read()
        except FileNotFoundError:
            print("Error: Could not find ecapa.yaml or paste.txt")
            sys.exit(1)

    # Override data_folder path
    manifest_dir = "manifests"
    for i, arg in enumerate(sys.argv):
        if arg == "--manifest_dir" and i + 1 < len(sys.argv):
            manifest_dir = sys.argv[i + 1]
    content = re.sub(
        r"(?m)^(\s*)data_folder:\s*\S+",
        rf"\1data_folder: {manifest_dir}",
        content,
    )

    # Disable multiprocessing
    content = re.sub(
        r"(?m)^(\s*)num_workers:\s*\d+",
        r"\1num_workers: 0",
        content,
    )

    # Desired STFT settings in samples
    n_fft_samples = 512
    win_len_samples = 512
    hop_len_samples = 128
    sample_rate = 16000

    # Convert to milliseconds for YAML
    win_len_ms = int(win_len_samples * 1000 / sample_rate)  # -> 32 ms
    hop_len_ms = int(hop_len_samples * 1000 / sample_rate)  # -> 8 ms

    # Override any STFT-related entries
    content = re.sub(
        r"(?m)^(\s*)n_fft:\s*\d+",
        rf"\1n_fft: {n_fft_samples}",
        content,
    )
    content = re.sub(
        r"(?m)^(\s*)win_length:\s*\d+",
        rf"\1win_length: {win_len_ms}",
        content,
    )
    content = re.sub(
        r"(?m)^(\s*)hop_length:\s*\d+",
        rf"\1hop_length: {hop_len_ms}",
        content,
    )

    # Write out the fixed YAML
    with open("temp_ecapa.yaml", "w") as fout:
        fout.write(content)

    # Run the training script
    cmd = [sys.executable, "ecapa.py", "temp_ecapa.yaml", "--device=cpu"]
    print("Running with fixed YAML parameters…")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running ecapa.py: {e}")
        print("Inspect temp_ecapa.yaml for any remaining mismatches.")
        sys.exit(1)

    print("Training complete!")


if __name__ == "__main__":
    main()
