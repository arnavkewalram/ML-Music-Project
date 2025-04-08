# ML-Music-Project
Converts audio to sheet music and other file types using ML

# Music Transcription System

A complete backend for a deep learning system that transcribes music audio into sheet music using Python and PyTorch.

## Features

### Input Support
- Accepts pre-recorded audio files (WAV/MP3)
- Supports live audio input (microphone)
- Batch processing of multiple audio files

### Model Design
- Custom deep learning model built from scratch using PyTorch
- Detects:
  - Onsets
  - Pitches
  - Durations
  - Velocity
  - Tempo
  - Time signatures
  - Polyphonic notes (multiple simultaneous notes)
  - Instruments

### Output Options
- Multiple output formats:
  - MIDI
  - MusicXML
  - PDF sheet music
  - LilyPond
  - MuseScore compatible formats
  - Piano roll visualization
  - Audio playback of transcribed notes
- Real-time visualization for live audio

## Project Structure

```
music_transcription/
├── model/                  # Model architecture, loss functions, training, evaluation
│   ├── architecture.py     # Neural network architecture
│   ├── losses.py           # Custom loss functions
│   ├── trainer.py          # Training pipeline
│   └── evaluation.py       # Evaluation metrics and testing
├── preprocessing/          # Audio and MIDI processing
│   ├── audio_processor.py  # Audio loading and feature extraction
│   ├── midi_processor.py   # MIDI handling and conversion
│   ├── data_loader.py      # Dataset loading utilities
│   └── dataset_preprocessor.py # MAESTRO dataset preprocessing
├── inference/              # Inference and output generation
│   ├── transcriber.py      # Audio transcription
│   └── format_converters.py # Output format conversion
├── utils/                  # Utility functions
├── scripts/                # Command-line scripts
│   ├── train.py            # Training script
│   └── transcribe.py       # Transcription script
├── configs/                # Configuration files
├── data/                   # Data directory
│   ├── maestro/            # MAESTRO dataset
│   └── preprocessed/       # Preprocessed data
├── output/                 # Output directory
├── checkpoints/            # Model checkpoints
└── logs/                   # Training logs
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/music-transcription.git
cd music-transcription
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install additional system dependencies:
```bash
# For audio processing
sudo apt-get install portaudio19-dev

# For PDF generation (optional)
sudo apt-get install lilypond

# For MuseScore output (optional)
sudo apt-get install musescore3
```

## Usage

### Training

1. Download and preprocess the MAESTRO dataset:
```bash
python -m scripts.train --download --preprocess --dataset_dir ./data/maestro --preprocessed_dir ./data/preprocessed
```

2. Train the model:
```bash
python -m scripts.train --preprocessed_dir ./data/preprocessed --checkpoint_dir ./checkpoints --log_dir ./logs
```

3. Advanced training options:
```bash
python -m scripts.train --preprocessed_dir ./data/preprocessed \
                       --batch_size 16 \
                       --num_epochs 50 \
                       --learning_rate 0.0005 \
                       --optimizer adamw \
                       --scheduler cosine \
                       --experiment_name my_experiment
```

### Transcription

1. Transcribe an audio file:
```bash
python -m scripts.transcribe --checkpoint ./checkpoints/best_model.pt --input ./path/to/audio.wav --output_dir ./output
```

2. Use real-time transcription:
```bash
python -m scripts.transcribe --checkpoint ./checkpoints/best_model.pt --real_time --visualize
```

3. List available audio devices:
```bash
python -m scripts.transcribe --checkpoint ./checkpoints/best_model.pt --list_devices
```

4. Specify output format:
```bash
python -m scripts.transcribe --checkpoint ./checkpoints/best_model.pt --input ./path/to/audio.wav --format musicxml
```

### Evaluation

Evaluate the model on the test set:
```bash
python -m model.evaluation --checkpoint ./checkpoints/best_model.pt --preprocessed_dir ./data/preprocessed
```

## Dataset

This system uses the [MAESTRO v3.0.0](https://magenta.tensorflow.org/datasets/maestro) dataset for training, which contains paired audio and MIDI recordings of piano performances.

## Model Architecture

The model architecture consists of:

1. **Feature Extraction**: CNN-based feature extraction from mel spectrograms
2. **Temporal Modeling**: Bidirectional LSTM with attention mechanism
3. **Task-specific Heads**:
   - Onset Detection Module
   - Pitch Detection Module
   - Duration Estimation Module
   - Tempo Detection Module
   - Time Signature Detection Module
   - Instrument Classification Module
   - Velocity Estimation Module

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [MAESTRO Dataset](https://magenta.tensorflow.org/datasets/maestro)
- [PyTorch](https://pytorch.org/)
- [Librosa](https://librosa.org/)
- [pretty_midi](https://github.com/craffel/pretty-midi)
- [music21](https://web.mit.edu/music21/)
