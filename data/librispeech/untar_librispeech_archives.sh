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
extract "$DATA_DIR"/dev-clean.tar.gz
extract "$DATA_DIR"/dev-other.tar.gz

# test data
extract "$DATA_DIR"/test-clean.tar.gz
extract "$DATA_DIR"/test-other.tar.gz

# train data
extract "$DATA_DIR"/train-clean-100.tar.gz
extract "$DATA_DIR"/train-clean-360.tar.gz
extract "$DATA_DIR"/train-other-500.tar.gz


# copy meta information files
cp "$LIBRISPEECH_EXTRACT_DIR"/LibriSpeech/BOOKS.TXT "$LIBRISPEECH_META_DIR"/_archive_books.txt
cp "$LIBRISPEECH_EXTRACT_DIR"/LibriSpeech/CHAPTERS.TXT "$LIBRISPEECH_META_DIR"/_archive_chapters.txt
cp "$LIBRISPEECH_EXTRACT_DIR"/LibriSpeech/SPEAKERS.TXT "$LIBRISPEECH_META_DIR"/_archive_speakers.txt