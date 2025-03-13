import numpy as np
import matplotlib.pyplot as plt
from openwakeword.model import Model
import scipy.io.wavfile
import argparse
import os
import mplcursors  # Import mplcursors for interactivity

# Parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_dirs",
    help="Comma-separated list of directories containing the WAV files to process",
    type=str,
    default="examples/audio/beta/no_back,examples/audio/beta/low_back,examples/audio/beta/med_back,examples/audio/beta/high_back",
    required=False
)
parser.add_argument(
    "--thresholds",
    help="Comma-separated list of thresholds to test",
    type=str,
    default="0.0125,0.025,0.075,0.1",
    required=False
)
parser.add_argument(
    "--vad_threshold",
    help="""The threshold to use for voice activity detection (VAD) in the openWakeWord instance.
            The default (0.0), disables VAD.""",
    type=float,
    default=0.0,
    required=False
)
parser.add_argument(
    "--noise_suppression",
    help="Whether to enable speex noise suppression in the openWakeWord instance.",
    type=bool,
    default=False,
    required=False
)
parser.add_argument(
    "--model_path",
    help="The path of a specific model to load",
    type=str,
    default="/Users/SAI/Documents/Code/openWakeWord2/wakeword_models/hey_zelda/hey_Zelda_cv11.onnx",
    required=False
)
parser.add_argument(
    "--inference_framework",
    help="The inference framework to use (either 'onnx' or 'tflite')",
    type=str,
    default='onnx',
    required=False
)

args = parser.parse_args()

# Convert threshold argument to list of floats
# thresholds = list(map(float, args.thresholds.split(',')))
# thresholds = [0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02, 0.025, 0.0375, 0.05, 0.0625, 0.075, 0.0875, 0.1, 0.1125, 0.125, 0.1375,]
thresholds = np.logspace(-2.5, 0, num=50)

patience = 1

# Load pre-trained openwakeword model
owwModel = Model(
    wakeword_models=[args.model_path], 
    enable_speex_noise_suppression=args.noise_suppression,
    vad_threshold=args.vad_threshold,
    inference_framework=args.inference_framework
)

# Warm-up the model with a dummy input
dummy_audio = np.zeros((5 * 16000, ), dtype=np.int16)  # 5 seconds of silence
for _ in range(10):  # Run 10 dummy inferences
    owwModel.predict(dummy_audio)

# Function to find the sequence of scores with the maximum sum of `patience` consecutive predictions
def max_sliding_window(scores, patience=3):
    max_sum_indices = (0, patience)
    max_sum = sum(scores[:patience])

    for i in range(1, len(scores) - patience + 1):
        window_sum = sum(scores[i:i + patience])
        if window_sum > max_sum:
            max_sum = window_sum
            max_sum_indices = (i, i + patience)
    
    return scores[max_sum_indices[0]:max_sum_indices[1]]

# Split the input directories
input_dirs = args.input_dirs.split(',')

# Prepare to store detection counts for each input directory
all_false_rejects = []

# Process each directory for false reject metrics
for input_dir in input_dirs:
    all_max_score_sequences = []
    total_files = 0
    print("Processing:", input_dir)

    for root, _, files in os.walk(input_dir):
        for filename in files:
            if filename.endswith(".wav"):
                total_files += 1
                filepath = os.path.join(root, filename)
                sample_rate, mic_audio = scipy.io.wavfile.read(filepath)

                # Slice the audio to the first 5 seconds
                five_second_length = 5 * sample_rate  # Number of samples in 5 seconds
                mic_audio = mic_audio[:five_second_length]

                # Feed to openWakeWord model and collect consecutive scores
                scores = []
                for i in range(0, len(mic_audio), 1280):  # Process in chunks of 1280 samples (80 ms at 16kHz)
                    chunk = mic_audio[i:i + 1280]
                    if len(chunk) < 1280:
                        break  # Skip if the chunk is less than 80 ms
                    prediction = owwModel.predict(chunk)
                    max_score = max(prediction.values())
                    scores.append(max_score)

                # Find the three consecutive scores with the highest sum
                max_score_sequence = max_sliding_window(scores, patience=patience)
                all_max_score_sequences.append(max_score_sequence)

    # Calculate false rejects for each threshold
    total_activations = len(all_max_score_sequences)
    if total_activations == 0:
        print(f"Warning: No activations found in {input_dir}. Check your data.")
        all_false_rejects.append([None] * len(thresholds))
        continue

    false_rejects = [
        (total_activations - sum(1 for score_seq in all_max_score_sequences if all(score >= threshold for score in score_seq))) 
        / total_activations * 100 for threshold in thresholds
    ]

    all_false_rejects.append(false_rejects)

# Define a variable for the plot name
model_name = os.path.basename(args.model_path)
plot_name = model_name + ' False Reject Rate vs Threshold'

# Plot False Reject Rate vs Threshold
fig, ax = plt.subplots(figsize=(10, 6))
lines = []
fig.canvas.manager.set_window_title(plot_name)

for idx, false_rejects in enumerate(all_false_rejects):
    dir_name = os.path.basename(input_dirs[idx].rstrip('/'))
    line, = ax.plot(thresholds, false_rejects, marker='o', label=dir_name)
    lines.append(line)

# Set up axes
ax.set_xlabel('Threshold')
ax.set_ylabel('False Reject Rate (%)')
ax.set_xscale('log')  # Log scale for better visualization
ax.set_title(plot_name)
ax.grid(True)

# Legend handling
ax.legend(lines, [line.get_label() for line in lines], bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.tight_layout(rect=[0, 0, 1, 1])  # Adjust layout to make room for the legend

# Create a cursor for interactive hovering
cursor = mplcursors.cursor(lines, hover=True)

# Customize hover annotations to show values
@cursor.connect("add")
def on_add(sel):
    x = sel.target[0]
    annotations = []
    for idx, false_rejects in enumerate(all_false_rejects):
        label = os.path.basename(input_dirs[idx].rstrip('/'))
        closest_idx = np.argmin(np.abs(np.array(thresholds) - x))
        y_value = false_rejects[closest_idx]
        annotations.append(f"{label}: {y_value:.1f}% False Reject Rate (Threshold: {thresholds[closest_idx]:.4f})")
    sel.annotation.set_text(f"Threshold = {x:.4f}\n" + "\n".join(annotations))

print(f"Generating plot: {plot_name}")
plt.show()
