#! /usr/bin/env bash
set -e

# check if download location is set
if [[ -z "$LIBRISPEECH_DOWNLOAD_DIR" ]]; then
  echo "Please set LIBRISPEECH_DOWNLOAD_DIR before calling this script"
  exit 1
fi

# useful for navigating to common scripts
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# make sure all potential directories exist
mkdir -p "$LIBRISPEECH_DOWNLOAD_DIR"

# default directory to save files in
DIR=$LIBRISPEECH_DOWNLOAD_DIR
echo "Downloading LibriSpeech dataset to $DIR"

## download files
echo "--- Downloading dev set ---"
echo "--- clean"
curl -C - https://www.openslr.org/resources/12/dev-clean.tar.gz --output "$DIR"/dev-clean.tar.gz
echo "--- other"
curl -C - https://www.openslr.org/resources/12/dev-other.tar.gz --output "$DIR"/dev-other.tar.gz

echo "--- Downloading test set ---"
echo "--- clean"
curl -C - https://www.openslr.org/resources/12/test-clean.tar.gz --output "$DIR"/test-clean.tar.gz
echo "--- other"
curl -C - https://www.openslr.org/resources/12/test-other.tar.gz --output "$DIR"/test-other.tar.gz

echo "--- Downloading train set ---"
echo "--- 100h"
curl -C - https://www.openslr.org/resources/12/train-clean-100.tar.gz --output "$DIR"/train-clean-100.tar.gz
echo "--- 360h"
curl -C - https://www.openslr.org/resources/12/train-clean-360.tar.gz --output "$DIR"/train-clean-360.tar.gz
echo "--- 500h"
curl -C - https://www.openslr.org/resources/12/train-other-500.tar.gz --output "$DIR"/train-other-500.tar.gz

