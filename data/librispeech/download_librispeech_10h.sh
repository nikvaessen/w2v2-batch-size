#! /usr/bin/env bash
set -e

# check if download location is set
if [[ -z "$LIBRISPEECH_DOWNLOAD_DIR" ]]; then
  echo "Please set LIBRISPEECH_DOWNLOAD_DIR before calling this script"
  exit 1
fi

# make sure all potential directories exist
mkdir -p "$LIBRISPEECH_DOWNLOAD_DIR"

# default directory to save files in
DIR=$LIBRISPEECH_DOWNLOAD_DIR
echo "Downloading LibriSpeech dataset to $DIR"

## download files
echo "--- Downloading 10h librispeech ---"
curl -C - https://dl.fbaipublicfiles.com/librilight/data/librispeech_finetuning.tgz --output "$DIR"/librispeech_10h.tar.gz

# verify checksums
echo "verifying checksums"
verify_checksum "$DIR"/librispeech_10h.tar.gz 7f83024cb1334bfa372d1af2c75c3a77 --algo md5
