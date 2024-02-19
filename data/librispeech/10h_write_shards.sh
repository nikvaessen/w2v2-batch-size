#! /usr/bin/env bash
set -e

# check whether paths have been set
if [[ -z "$LIBRISPEECH_SHARD_DIR" ]]; then
  echo "Please set LIBRISPEECH_SHARD_DIR before calling this script"
  exit 3
fi

# make sure all potential directories exist
mkdir -p "$LIBRISPEECH_SHARD_DIR"

# folders for each split
train_10m=$LIBRISPEECH_SHARD_DIR/train-10m
train_1h=$LIBRISPEECH_SHARD_DIR/train-1h
train_10h=$LIBRISPEECH_SHARD_DIR/train-10h

# set the number of parallel processes writing shards to disk
workers=2

# make sure there's no previous tar files
rm -f "$LIBRISPEECH_SHARD_DIR"/train_10m/*.tar*
rm -f "$LIBRISPEECH_SHARD_DIR"/train_1h/*.tar*
rm -f "$LIBRISPEECH_SHARD_DIR"/train_10h/*.tar*

# write the train shards
write_tar_shards \
--csv "$train_10m"/_meta.train_10m.csv \
--out "$train_10m" \
--strategy length_sorted \
--samples_per_shard 5000 \
--prefix train_10m \
--workers="$workers"

write_tar_shards \
--csv "$train_1h"/_meta.train_1h.csv \
--out "$train_1h" \
--strategy length_sorted \
--samples_per_shard 5000 \
--prefix train_1h \
--workers="$workers"

write_tar_shards \
--csv "$train_10h"/_meta.train_10h.csv \
--out "$train_10h" \
--strategy length_sorted \
--samples_per_shard 5000 \
--prefix train_10h \
--workers="$workers"
