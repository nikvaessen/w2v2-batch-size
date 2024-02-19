#! /usr/bin/env bash
set -e

# check whether paths have been set
if [[ -z "$LIBRISPEECH_EXTRACT_DIR" ]]; then
  echo "Please set LIBRISPEECH_EXTRACT_DIR before calling this script"
  exit 2
fi

if [[ -z "$LIBRISPEECH_SHARD_DIR" ]]; then
  echo "Please set LIBRISPEECH_SHARD_DIR before calling this script"
  exit 3
fi

if [[ -z "$LIBRISPEECH_META_DIR" ]]; then
  echo "Please set LIBRISPEECH_META_DIR before calling this script"
  exit 4
fi

# make sure all potential directories exist
mkdir -p "$LIBRISPEECH_EXTRACT_DIR" "$LIBRISPEECH_SHARD_DIR" "$LIBRISPEECH_META_DIR"

# useful for navigating to common scripts
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# folders for each split
train_10m=$LIBRISPEECH_SHARD_DIR/train-10m
train_1h=$LIBRISPEECH_SHARD_DIR/train-1h
train_10h=$LIBRISPEECH_SHARD_DIR/train-10h

# make folders
mkdir -p "$train_10m" "$train_1h" "$train_10h"

# make the CSV file for each of the 3 train splits
"$SCRIPT_DIR"/generate_csv.py \
--dir "$LIBRISPEECH_EXTRACT_DIR"/LibriSpeech/train-10m \
--csv "$train_10m"/_meta.train_10m.csv \
--speakers "$LIBRISPEECH_META_DIR"/_archive_speakers.txt \
--ext wav

"$SCRIPT_DIR"/generate_csv.py \
--dir "$LIBRISPEECH_EXTRACT_DIR"/LibriSpeech/train-1h \
--csv "$train_1h"/_meta.train_1h.csv \
--speakers "$LIBRISPEECH_META_DIR"/_archive_speakers.txt \
--ext wav

"$SCRIPT_DIR"/generate_csv.py \
--dir "$LIBRISPEECH_EXTRACT_DIR"/LibriSpeech/train-10h \
--csv "$train_10h"/_meta.train_10h.csv \
--speakers "$LIBRISPEECH_META_DIR"/_archive_speakers.txt \
--ext wav



