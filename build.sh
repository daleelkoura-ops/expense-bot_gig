#!/bin/bash
echo "🔧 Build script starting..."

# Download Vosk model if it doesn't exist
if [ ! -d "vosk-model-en-us-0.22-lgraph" ]; then
    echo "📥 Downloading Vosk model..."
    wget -q https://alphacephei.com/vosk/models/vosk-model-en-us-0.22-lgraph.zip -O model.zip
    echo "📦 Extracting model..."
    unzip -q model.zip
    rm model.zip
    echo "✅ Vosk model ready!"
else
    echo "✅ Vosk model already exists, skipping download."
fi