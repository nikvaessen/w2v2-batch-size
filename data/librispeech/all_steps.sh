#! /usr/bin/env bash
set -e

# check all environment variables
if [[ -z "$LIBRISPEECH_DOWNLOAD_DIR" ]]; then
  echo "Please set LIBRISPEECH_DOWNLOAD_DIR before calling this script"
  exit 1
fi

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

# if NUM_CPU is not set, use all of them by default
if [ -z ${NUM_CPU+x} ]; then
  NUM_CPU=$(nproc) # or fewer if you want to use the PC for other stuff...
fi

# make sure all potential directories exist
mkdir -p "$LIBRISPEECH_DOWNLOAD_DIR" "$LIBRISPEECH_EXTRACT_DIR" \
"$LIBRISPEECH_SHARD_DIR" "$LIBRISPEECH_META_DIR"

echo "LIBRISPEECH_DOWNLOAD_DIR=$LIBRISPEECH_DOWNLOAD_DIR"
echo "LIBRISPEECH_EXTRACT_DIR=$LIBRISPEECH_EXTRACT_DIR"
echo "LIBRISPEECH_SHARD_DIR=$LIBRISPEECH_SHARD_DIR"
echo "LIBRISPEECH_META_DIR=$LIBRISPEECH_META_DIR"
echo "NUM_CPU=$NUM_CPU"

# check ffmpeg is installed
if ! [ -x "$(command -v ffmpeg)" ]; then
  echo 'Error: ffmpeg is not installed.' >&2
  exit 5
fi

# move to folder of this script so path to all scripts are known
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR" || exit

# execute ALL steps in order :)

# download
./download_librispeech.sh
./download_librispeech_10h.sh

# extract
./untar_librispeech_archives.sh
./untar_librispeech_10h.sh

# convert to wav
convert_to_wav \
  --dir "$LIBRISPEECH_EXTRACT_DIR"/LibriSpeech \
  --ext .flac \
  --workers "$NUM_CPU"

# write default shards (full 960h)
./default_setup_shards.sh
./default_write_shards.sh

# write small shards (10m, 1h, 10h)
./10h_setup_shards.sh
./10h_write_shards.sh
