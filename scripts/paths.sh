#!/usr/bin/env bash
# paths.sh — Central path definitions for voxtral.c scripts
# Source this file: . "$(dirname "$0")/../paths.sh"  (from subfolder)
#                   . "$(dirname "$0")/paths.sh"      (from scripts/)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Model
MODEL_DIR="${MODEL_DIR:-$REPO_ROOT/voxtral-model}"
MODEL_ID="mistralai/Voxtral-Mini-4B-Realtime-2602"

# Binary
VOXTRAL_BIN="${VOXTRAL_BIN:-$REPO_ROOT/voxtral}"

# Samples
SAMPLES_DIR="$REPO_ROOT/samples"
TEST_SAMPLE="$SAMPLES_DIR/test_speech.wav"
ITALIAN_SAMPLE="$SAMPLES_DIR/antirez_speaking_italian_short.ogg"
JFK_SAMPLE="$SAMPLES_DIR/jfk.wav"
DREAM_SAMPLE="$SAMPLES_DIR/I_have_a_dream.ogg"

# CUDA
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
