##################################

# This example scripts runs openWakeWord continuously on a microphone stream,
# and saves 5 seconds of audio immediately before the activation as WAV clips
# in the specified output location.

##################################

# Imports
import os
import sys
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import platform
import collections
import time
if platform.system() == "Windows":
    import pyaudiowpatch as pyaudio
else:
    import pyaudio
import numpy as np
from openwakeword.model import Model
import openwakeword
import scipy.io.wavfile
import datetime
import argparse

# Parse input arguments
parser=argparse.ArgumentParser()
parser.add_argument(
    "--output_dir",
    help="Where to save the audio that resulted in an activation",
    type=str,
    default="positivedetections",
    required=False
)
parser.add_argument(
    "--threshold",
    help="The score threshold for an activation",
    type=float,
    default=0.18,
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
# parser=argparse.ArgumentParser()
parser.add_argument(
    "--chunk_size",
    help="How much audio (in number of 16khz samples) to predict on at once",
    type=int,
    default=1280,
    required=False
)
parser.add_argument(
    "--model_path",
    help="The path of a specific model to load",
    type=str,
    default="wakeword_models/hey_zelda/hey_Zelda_8_15.onnx",
    required=False
)
parser.add_argument(
    "--inference_framework",
    help="The inference framework to use (either 'onnx' or 'tflite'",
    type=str,
    default='onnx',
    required=False
)
parser.add_argument(
    "--disable_activation_sound",
    help="Disables the activation sound, clips are silently captured",
    action='store_true',
    required=False
)

args=parser.parse_args()

# Get microphone stream
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = args.chunk_size * 4
audio = pyaudio.PyAudio()
mic_stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)


# Load pre-trained openwakeword models
if args.model_path != "":
    owwModel = Model(
        wakeword_models=[args.model_path], 
        enable_speex_noise_suppression=args.noise_suppression,
        vad_threshold = args.vad_threshold,
        inference_framework=args.inference_framework
        )
else:
    owwModel = Model(inference_framework=args.inference_framework)

# Set waiting period after activation before saving clip (to get some audio context after the activation)
save_delay = 1  # seconds

# Set cooldown period before another clip can be saved
cooldown = 4  # seconds

# Create output directory if it does not already exist
if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)

# Run capture loop, checking for hotwords
if __name__ == "__main__":
    # Predict continuously on audio stream
    last_save = time.time()
    activation_times = collections.defaultdict(list)

    print("\n\nListening for wakewords...\n")
    while True:
        # Get audio
        mic_audio = np.frombuffer(mic_stream.read(CHUNK), dtype=np.int16)
        patience = {}  # e.g., {"model_name": 3} if using patience, or leave as empty if not
        threshold = {"hey_Zelda_8_15": 0.18}  # Replace "hey_Zelda_model" with the actual model name in your setup

        # Feed to openWakeWord model
        prediction = owwModel.predict(mic_audio, debounce_time=1.25, threshold=threshold)

        # Check for model activations (score above threshold), and save clips
        for mdl in prediction.keys():
            if prediction[mdl] >= args.threshold:
                activation_times[mdl].append(time.time())

            if activation_times.get(mdl) and (time.time() - last_save) >= cooldown \
               and (time.time() - activation_times.get(mdl)[0]) >= save_delay:
                last_save = time.time()
                activation_times[mdl] = []
                detect_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                
                print(f'Detected activation from \"{mdl}\" model at time {detect_time}!')

                # Capture total of 5 seconds, with the microphone audio associated with the
                # activation around the ~4 second point
                audio_context = np.array(list(owwModel.preprocessor.raw_data_buffer)[-16000*5:]).astype(np.int16)
                fname = detect_time + f"_{mdl}.wav"
                scipy.io.wavfile.write(os.path.join(os.path.abspath(args.output_dir), fname), 16000, audio_context)
                
