![Github CI](https://github.com/dscripka/openWakeWord/actions/workflows/tests.yml/badge.svg)

# openWakeWord

openWakeWord is an open-source wakeword library that can be used to create voice-enabled applications and interfaces. It includes pre-trained models for common words & phrases that work well in real-world environments.

**Quick Links**
- [Installation](#installation)
- [Training New Models](#training-new-models)
- [FAQ](#faq)

# Updates

**2024/02/11**
- v0.6.0 of openWakeWord released. See the [releases](https://github.com/dscripka/openWakeWord/releases) for a full descriptions of new features and changes.

**2023/11/09**
- Added example scripts under `examples/web` that demonstrate streaming audio from a web application into openWakeWord.

**2023/10/11**
- Significant improvements to the process of [training new models](#training-new-models), including an example Google Colab notebook demonstrating how to train a basic wake word model in <1 hour.

**2023/06/15**
- v0.5.0 of openWakeWord released. See the [releases](https://github.com/dscripka/openWakeWord/releases) for a full descriptions of new features and changes.

# Demo

You can try an online demo of the included pre-trained models via HuggingFace Spaces [right here](https://huggingface.co/spaces/davidscripka/openWakeWord)!

Note that real-time detection of a microphone stream can occasionally behave strangely in Spaces. For the most reliable testing, perform a local installation as described below.

# Installation

Installing openWakeWord is simple and has minimal dependencies:

```
pip install openwakeword
```

On Linux systems, both the [onnxruntime](https://pypi.org/project/onnxruntime/) package and [tflite-runtime](https://pypi.org/project/tflite-runtime/) packages will be installed as dependencies since both inference frameworks are supported. On Windows, only onnxruntime is installed due to a lack of support for modern versions of tflite.


# Usage

For quick local testing, clone this repository and use the included [example script](examples/detect_from_microphone.py) to try streaming detection from a local microphone. You can individually download pre-trained models from current and past [releases](https://github.com/dscripka/openWakeWord/releases/), or you can download them using Python (see below).

Adding openWakeWord to your own Python code requires just a few lines:

```python
import openwakeword
from openwakeword.model import Model

# Instantiate the model(s)
model = Model(
    wakeword_models=["path/to/model.tflite"],  # can also leave this argument empty to load all of the included pre-trained models
)

# Get audio data containing 16-bit 16khz PCM audio data from a file, microphone, network stream, etc.
# For the best efficiency and latency, audio frames should be multiples of 80 ms, with longer frames
# increasing overall efficiency at the cost of detection latency
frame = my_function_to_get_audio_frame()

# Get predictions for the frame
prediction = model.predict(frame)
```

Additionally, openWakeWord provides other useful utility functions. For example:

```python
# Get predictions for individual WAV files (16-bit 16khz PCM)
from openwakeword.model import Model

model = Model()
model.predict_clip("path/to/wav/file")

# Get predictions for a large number of files using multiprocessing
from openwakeword.utils import bulk_predict

bulk_predict(
    file_paths = ["path/to/wav/file/1", "path/to/wav/file/2"],
    wakeword_models = ["hey zelda"],
    ncpu=2
)
```

See `openwakeword/utils.py` and `openwakeword/model.py` for the full specification of class methods and utility functions.

# Model Architecture

openWakeword models are composed of three separate components:

1) A pre-processing function that computes [melspectrogram](https://pytorch.org/audio/main/generated/torchaudio.transforms.MelSpectrogram.html) of the input audio data. For openWakeword, an ONNX implementation of Torch's melspectrogram function with fixed parameters is used to enable efficient performance across devices.

2) A shared feature extraction backbone model that converts melspectrogram inputs into general-purpose speech audio embeddings. This [model](https://arxiv.org/abs/2002.01322) is provided by [Google](https://tfhub.dev/google/speech_embedding/1) as a TFHub module under an [Apache-2.0](https://opensource.org/licenses/Apache-2.0) license. For openWakeWord, this model was manually re-implemented to separate out different functionality and allow for more control of architecture modifications compared to a TFHub module. The model itself is series of relatively simple convolutional blocks, and gains its strong performance from extensive pre-training on large amounts of data. This model is the core component of openWakeWord, and enables the strong performance that is seen even when training on fully-synthetic data.

3) A classification model that follows the shared (and frozen) feature extraction model. The structure of this classification model is arbitrary, but in practice a simple fully-connected network or 2 layer RNN works well.

# Performance and Evaluation

Evaluating wake word/phrase detection models is challenging, and it is often very difficult to assess how different models presented in papers or other projects will perform *when deployed* with respect to two critical metrics: false-reject rates and false-accept rates. For clarity in definitions:

A *false-reject* is when the model fails to detect an intended activation from a user.

A *false-accept* is when the model inadvertently activates when the user did not intend for it to do so.

For openWakeWord, evaluation follows two principles:

- The *false-reject* rate should be determined from wakeword/phrases that represent realistic recording environments, including those with background noise and reverberation. This can be accomplished by directly collected data from these environments, or simulating them with data augmentation methods.

- The *false-accept* rate should be determined from audio that represents the types of environments that would be expected for the deployed model, not just on the training/evaluation data. In practice, this means that the model should only rarely activate in error, even in the presence of hours of continuous speech and background noise.


## Other Performance Details

### Model Robustness

Due to a combination of variability in the generated speech and the extensive pre-training from Google, openWakeWord models also demonstrate some additional performance benefits that are useful for real-world applications. In testing, three in particular have been observed.

1) The trained models seem to respond reasonably well to wakewords and phrases that are [whispered](https://en.wikipedia.org/wiki/Whispering). This is somewhat surprising behavior, as the text-to-speech models used for producing training data generally do not create synthetic speech that has acoustic qualities similar to whispering.

2) The models also respond relatively well to wakewords and phrases spoken at different speeds (within reason).

3) The models are able to handle some variability in the phrasing of a given command. This behavior was not entirely a surprise, given that [others](https://arxiv.org/abs/1904.03670) have reported similar benefits when training end-to-end spoken language understanding systems. For example, the included [pre-trained weather model](docs/models/weather.md) will typically still respond correctly to a phrase like "how is the weather today" despite not training directly on that phrase (though false rejections rates will likely be higher, on average, compared to phrases closer to the training data).

### Background Noise

While the models are trained with background noise to increase robustness, in some cases additional noise suppression can improve performance. Setting the `enable_speex_noise_suppression=True` argument during openWakeWord model initialization will use the efficient Speex noise suppression algorithm to pre-process the audio data prior to prediction. This can reduce both false-reject rates and false-accept rates, though testing in a realistic deployment environment is strongly recommended.

# Training New Models

openWakeWord includes an automated utility that greatly simplifies the process of training custom models. This can be used in two ways:

1) A simple [Google Colab](https://colab.research.google.com/drive/1q1oe2zOyZp7UsB3jJiQ1IFn8z5YfjwEb?usp=sharing) notebook with an easy to use interface and simple end-to-end process. This allows anyone to produce a custom model very quickly (<1 hour) and doesn't require any development experience, but the performance of the model may be low in some deployment scenarios.

2) A more detailed [notebook](notebooks/automatic_model_training.ipynb) (also on [Google Colab](https://colab.research.google.com/drive/1yyFH-fpguX2BTAW8wSQxTrJnJTM-0QAd?usp=sharing)) that describes the training process in more details, and enables more customization. This can produce high quality models, but requires more development experience.

For a collection of models trained using the notebooks above by the Home Assistant Community (and with much gratitude to @fwartner), see the excellent repository [here](https://github.com/fwartner/home-assistant-wakewords-collection).

For users interested in understanding the fundamental concepts behind model training there is a more detailed, educational [tutorial notebook](notebooks/training_models.ipynb) also available. However, this specific notebook is not intended for training production models, and the automated process above is recommended for that purpose.

Fundamentally, a new model requires two data generation and collection steps:

1) Generate new training data for the desired wakeword/phrase using open-source speech-to-text systems (see [Synthetic Data Generation](docs/synthetic_data_generation.md) for more details). These models and the generation code are hosted in a separate [repository](https://github.com/dscripka/synthetic_speech_dataset_generation). The number of generated examples required can vary, a minimum of several thousand is recommended and performance seems to increase smoothly with increasing dataset size.

2) Collect negative data (e.g., audio where the wakeword/phrase is not present) to help the model have a low false-accept rate. This also benefits from scale, and the [included models](#pre-trained-models) were all trained with ~30,000 hours of negative data representing speech, noise, and music. See the individual model documentation pages for more details on training data curation and preparation.

# Language Support

Currently, openWakeWord only supports English, primarily because the pre-trained text-to-speech models used to generate training data are all based on english datasets. It's likely that speech-to-text models trained on other languages would also work well, but non-english models & datasets are less commonly available.

Future release road maps may have non-english support. In particular, [Mycroft.AIs Mimic 3](https://github.com/MycroftAI/mimic3-voices) TTS engine may work well to help extend some support to other languages.

# FAQ

**Why are there three separate models instead of just one?**
- Separating the models was an intentional choice to provide flexibility and optimize the efficiency of the end-to-end prediction process. For example, with separate melspectrogram, embedding, and prediction models, each one can operate on different size inputs of audio to optimize overall latency and share computations between models. It certainly is possible to make a combined model with all of the steps integrated, though, if that was a requirement of a particular use case.

# Acknowledgements

I am very grateful for the encouraging and positive response from the open-source community since the release of openWakeWord in January 2023. In particular, I want to acknowledge and thank the following individuals and groups for their feedback, collaboration, and development support:

- [synesthesiam](https://github.com/synesthesiam)
- [SecretSauceAI](https://github.com/secretsauceai)
- [OpenVoiceOS](https://github.com/OpenVoiceOS)
- [Nabu Casa](https://github.com/NabuCasa)
- [Home Assistant](https://github.com/home-assistant)

# License

All of the code in this repository is licensed under the **Apache 2.0** license. All of the included pre-trained models are licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/) license due to the inclusion of datasets with unknown or restrictive licensing as part of the training data. If you are interested in pre-trained models with more permissive licensing, please raise an issue and we will try to add them to a future release.
