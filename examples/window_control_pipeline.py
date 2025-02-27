##################################

# This example scripts runs openWakeWord continuously on a microphone stream,
# and saves 5 seconds of audio immediately before the activation as WAV clips
# in the specified output location.

##################################

# Imports
import os
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
from utils.beep import playBeep
from utils.capture_detections import cos_sim, compute_similarity2, compute_similarity_single_frame, compute_similarity_single_slice
from utils.convert_audio import convert_and_augment, convert_and_augment_tensor
from utils.speaker_embedding import generate_speaker_embedding_tensor
import torch
from resemblyzer import VoiceEncoder
import time
from pydub import AudioSegment

# Parse input arguments
parser=argparse.ArgumentParser()
parser.add_argument(
    "--output_dir",
    help="Where to save the audio that resulted in an activation",
    type=str,
    default="positivedetections/meeting_test/aug_1_standup//hey_zelda_multi_phrase",
    required=False
)
parser.add_argument(
    "--threshold",
    help="The score threshold for an activation",
    type=float,
    default=0.5,
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
    default="/Users/SAI/Documents/Code/wakeWord/wakeWordForked/Untitled/wakeword_models/hey_zelda/hey_Zelda_med_multi_phrase.onnx",
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

def int2float(sound):
    abs_max = np.abs(sound).max()
    sound = sound.astype('float32')
    if abs_max > 0:
        sound *= 1/32768
    sound = sound.squeeze()  # depends on the use case
    return sound

model, utils2 = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=True)

speaker_encoder = VoiceEncoder()
speaker_encoder.eval()
(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils2

#warmup
warmup_file1 = '/Users/SAI/Documents/Code/wakeWord/wakeWordForked/Untitled/examples/audio/hey_zelda/no_mask/no_back/epp_no_background_8_19_stand_54db/epp_stand_nobckgrn_8_1923.wav'
audiotens = convert_and_augment(warmup_file1, "examples/audio/augments", "examples/audio/clean_augmentations/noise", "examples/audio/clean_augmentations/rir", augment=True, save=True)
speaker_embedding2 = generate_speaker_embedding_tensor(audiotens, speaker_encoder)

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
save_delay = 0.5  # seconds

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
    detected_once = False  # Flag to check if detection happened once

    print("\n\nListening for wakewords...\n")
    while not detected_once:  # Main loop to listen until first detection
        # Get audio
        mic_audio = np.frombuffer(mic_stream.read(CHUNK), dtype=np.int16)

        # Feed to openWakeWord model
        prediction = owwModel.predict(mic_audio)

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
                detected_once = True  # Set flag to True after first detection

                # Capture total of 5 seconds, with the microphone audio
                audio_context = np.array(list(owwModel.preprocessor.raw_data_buffer)[-16000*4:]).astype(np.int16)

                fname = detect_time + f"_{mdl}.wav"
                saved_file_path = os.path.join(os.path.abspath(args.output_dir), fname)
                scipy.io.wavfile.write(saved_file_path, 16000, audio_context)

                # Convert augmented audio to tensor
                audiotens = convert_and_augment(saved_file_path, "examples/audio/augments", "examples/audio/clean_augmentations/noise", "examples/audio/clean_augmentations/rir", augment=True, save=False)
                speaker_embedding = generate_speaker_embedding_tensor(audiotens, speaker_encoder)

                # if not args.disable_activation_sound:
                #     playBeep(os.path.join(os.path.dirname(__file__), 'audio', 'activation.wav'), audio)

    # New loop after first positive detection

    # Initialize a rolling buffer for the last 1 second of audio data
    from collections import deque
    BUFFER_SECONDS = 1  # We want the last 1 second of audio

    # Initialize deque buffer with a fixed length
    # audio_buffer = deque(maxlen=RATE * BUFFER_SECONDS)
    audio_buffer = deque(maxlen=int(RATE * BUFFER_SECONDS))

    # Initialize deque to store the last 5 similarity scores
    similarity_scores = deque(maxlen=5)

    try:
        print("Starting audio stream for similarity comparison...")
        while True:
            # Read chunk of audio data
            audio_data = mic_stream.read(CHUNK)
            
            # Convert to NumPy array and normalize
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Append new audio chunk to the buffer
            audio_buffer.extend(audio_np)
            
            # Only process once we have 1 second of audio in the buffer
            if len(audio_buffer) == RATE * BUFFER_SECONDS:
                # Convert buffer to a numpy array and pass it as the slice
                buffer_np = np.array(audio_buffer)
                
                audio_int16 = (buffer_np * 32768).astype(np.int16)  # Convert back to int16
                audio_float32 = int2float(audio_int16)
                audio_float32 = audio_float32[:512]
                audio_tensor = torch.from_numpy(audio_float32)
                
                # Get VAD confidence score
                vad_confidence = model(audio_tensor, RATE).item()
                
                # Compute similarity score for this slice using the new function
                similarity_score = compute_similarity_single_slice(buffer_np, speaker_embedding, speaker_encoder).item()
                score_product = similarity_score * vad_confidence
                
                print(f"Product of Similarity and VAD Confidence Scores: {score_product} {vad_confidence}")

                # Store the similarity score in the deque
                # similarity_scores.append(similarity_score.item())
                
                # # Calculate the average of the last 5 similarity scores
                # if len(similarity_scores) == 5:
                #     avg_similarity = sum(similarity_scores) / len(similarity_scores)
                #     print(f"Average similarity score (last 5): {avg_similarity}")
                # else:
                #     print(f"Similarity score: {similarity_score.item()}")
            
            # Add a smaller sleep interval to help maintain real-time processing
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("Audio streaming stopped.")
    finally:
        # Close the stream and terminate the PyAudio instance
        mic_stream.stop_stream()
        mic_stream.close()
        audio.terminate()

