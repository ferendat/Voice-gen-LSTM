# Rhythmic LSTM Audio Generator

Generates rhythmic robotic audio using PyTorch LSTM networks. Creates 8-second audio clips with structured beat patterns and tonal variations.

## Features

- **LSTM-based audio synthesis** with 2-layer architecture
- **Rhythmic pattern generation** using 4/4 time signature
- **Multi-tonal beats** with frequency modulation and decay envelopes
- **GPU acceleration** support for fast training
- **WAV output** at 16kHz sample rate

## Requirements

```
torch
numpy
scipy
```

## Usage

```bash
python rhythmic_lstm_audio.py
```

The script will:
1. Generate training data with rhythmic patterns
2. Train LSTM model on sequential audio prediction
3. Generate 8 seconds of new rhythmic audio
4. Save output as `rhythmic_lstm_voice.wav`

## Technical Details

- **Model**: 2-layer LSTM with 128 hidden units
- **Sequence length**: 200 samples for temporal context
- **Beat patterns**: 4 distinct types (110Hz, 220Hz, modulated 165Hz, noise)
- **Training**: 20 epochs with Adam optimizer
- **Output**: 8-second monophonic audio at 16kHz

## Audio Characteristics

The generated audio features:
- Structured rhythmic patterns in 4/4 time
- Robotic/synthetic tonal quality
- Beat emphasis and decay envelopes
- Procedural variations learned from training data

## Hardware

Optimized for CUDA-enabled GPUs. Falls back to CPU if GPU unavailable.
