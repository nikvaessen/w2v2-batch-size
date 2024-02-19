#! /usr/bin/env bash
set -e

# check whether paths have been set
if [[ -z "$LIBRISPEECH_DOWNLOAD_DIR" ]]; then
  echo "Please set LIBRISPEECH_DOWNLOAD_DIR before calling this script"
  exit 1
fi

if [[ -z "$LIBRISPEECH_EXTRACT_DIR" ]]; then
  echo "Please set LIBRISPEECH_EXTRACT_DIR before calling this script"
  exit 2
fi

if [[ -z "$LIBRISPEECH_META_DIR" ]]; then
  echo "Please set LIBRISPEECH_META_DIR before calling this script"
  exit 4
fi

# useful for navigating to common scripts
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# make sure all potential directories exist
mkdir -p "$LIBRISPEECH_DOWNLOAD_DIR" "$LIBRISPEECH_EXTRACT_DIR" "$LIBRISPEECH_META_DIR"

# define paths to tar files and output location
DATA_DIR=$LIBRISPEECH_DOWNLOAD_DIR
EXTRACT_DIR=$LIBRISPEECH_EXTRACT_DIR

# make sure extraction dir exists
mkdir -p "$EXTRACT_DIR"

# utility function for extracting
extract() {
  tar_file=$1
  if [ -f "$tar_file" ]; then
    echo "extracting $tar_file"
    tar xzfv "$tar_file" -C "$EXTRACT_DIR" > /dev/null
  else
     echo "file $tar_file does not exist."
  fi
}

# dev data
extract "$DATA_DIR"/librispeech_10h.tar.gz

# make a new folder for each set which mimics data structure of librispeech
mkdir -p "$EXTRACT_DIR"/LibriSpeech/train-10m
mkdir -p "$EXTRACT_DIR"/LibriSpeech/train-1h
mkdir -p "$EXTRACT_DIR"/LibriSpeech/train-10h

# 10m
rsync "$EXTRACT_DIR"/librispeech_finetuning/1h/0/*/ "$EXTRACT_DIR"/LibriSpeech/train-10m/ -a

# 1h
rsync "$EXTRACT_DIR"/librispeech_finetuning/1h/*/*/ "$EXTRACT_DIR"/LibriSpeech/train-1h/ -a

# 10h
rsync "$EXTRACT_DIR"/librispeech_finetuning/1h/*/*/ "$EXTRACT_DIR"/LibriSpeech/train-10h/ -a
rsync "$EXTRACT_DIR"/librispeech_finetuning/9h/*/ "$EXTRACT_DIR"/LibriSpeech/train-10h/ -a

# fixes the transcription files of sessions which are in both 1h and 9h split
# (and thus would be overwritten by rsync)
python3 "$SCRIPT_DIR"/fix_txt_10h.py \
"$EXTRACT_DIR"/LibriSpeech/train-10h/ "$EXTRACT_DIR"/librispeech_finetuning/
