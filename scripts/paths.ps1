# paths.ps1 — Central path definitions for voxtral.c scripts
# Dot-source: . "$PSScriptRoot\..\paths.ps1"

$RepoRoot = (Resolve-Path "$PSScriptRoot\..").Path

# Model
if (-not $ModelDir) { $ModelDir = "$RepoRoot\voxtral-model" }
$ModelId = "mistralai/Voxtral-Mini-4B-Realtime-2602"

# Binary
if (-not $Voxtral) { $Voxtral = "$RepoRoot\voxtral.exe" }

# Samples
$SamplesDir = "$RepoRoot\samples"
$TestSample = "$SamplesDir\test_speech.wav"
$JfkSample = "$SamplesDir\jfk.wav"

# CUDA
if (-not $env:CUDA_PATH) { $env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0" }
