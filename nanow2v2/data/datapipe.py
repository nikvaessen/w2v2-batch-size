"""
Implement a datapipe based on a speech dataset saved in tar-based shards.
"""
import functools
import json
import fnmatch

from typing import Tuple

import torch
import torchaudio
from torch.utils.data import DataLoader

from torch.utils.data.datapipes.utils.common import StreamWrapper
from torchdata.datapipes.iter import (
    FileLister,
    FileOpener,
    TarArchiveLoader,
    Mapper,
    WebDataset,
    Shuffler,
    ShardingFilter,
    Header,
    IterDataPipe,
    MaxTokenBucketizer,
    Batcher,
    Cycler,
    Filter,
)

import pathlib
from dataclasses import dataclass
from typing import Optional, Dict, Union, List

from nanow2v2.data.encode import encode_transcription
from nanow2v2.data.pad import collate_append_constant

torchaudio.set_audio_backend("soundfile")

########################################################################################
# Creation of datapipe


@dataclass
class DataPipeBuilderConfig:
    # potential compression of shards
    tar_read_mode: str  # depends on compression or not

    # parameters determining randomness
    buffer_samples_before_batch: int
    buffer_batches: int

    # determine the size of buffer from which batches of almost equal length are drawn
    bucket_buffer: int

    # logic for giving each worker equal number of data
    allow_partial_shards: bool
    num_workers: int
    drop_last: Optional[bool] = None  # must be defined is batch_size is defined

    # crop audio to a particular length before batching
    crop_audio_frames: Optional[int] = None

    # batching (one must be defined)
    max_tokens: Optional[int] = None
    batch_size: Optional[int] = None  # useful for a test batch size of 1

    # bounds on length of data
    max_audio_frames: Optional[int] = None  # maximum length of a possible sample
    max_transcription_frames: Optional[int] = None  # maximum length of gt
    max_audio_frame_difference: Optional[
        int
    ] = None  # max difference between longest and shortest sequence in batch

    # whether to create an infinite dataloader
    make_infinite: bool = False

    # whether to run in debug mode, which limits the pipe to 10 samples.
    run_debug_mode: bool = False

    # whether to pin memory
    pin_memory: bool = False


class DataPipeBuilder:
    def __init__(
        self,
        cfg: DataPipeBuilderConfig,
        char_to_idx: Optional[Dict] = None,
        shard_workers: bool = True,
    ):
        self.cfg = cfg
        self.char_to_idx = char_to_idx
        self.shard_workers = shard_workers

    def get_pipe(
        self,
        shard_dirs: Union[str, List[str]],
        shard_file_pattern: str,
    ):
        # First get a stream of WavAudioSamples
        dp, shard_list = load_samples_from_shards(
            dirs_or_files=shard_dirs,
            shard_file_pattern=shard_file_pattern,
            tar_read_mode=self.cfg.tar_read_mode,
            allow_partial=self.cfg.allow_partial_shards,
            sample_buffer=self.cfg.buffer_samples_before_batch,
            create_infinite_datapipe=self.cfg.make_infinite,
            debug_mode=self.cfg.run_debug_mode,
            shard_workers=self.shard_workers,
        )

        if self.cfg.crop_audio_frames is not None:
            dp = Mapper(
                dp,
                functools.partial(
                    crop_data_sample, crop_size=self.cfg.crop_audio_frames
                ),
            )

        # convert to batches
        dp = map_to_batch(
            dp,
            char_to_idx=self.char_to_idx,
            max_tokens=self.cfg.max_tokens,
            batch_size=self.cfg.batch_size,
            drop_last=self.cfg.drop_last,
            max_len=self.cfg.max_audio_frames,
            buffer_size=self.cfg.bucket_buffer,
        )

        # filter batches where difference between length of samples is too large
        if self.cfg.max_audio_frame_difference is not None:
            dp = Filter(
                dp,
                functools.partial(
                    _filter_on_audio_frame_diff,
                    threshold=self.cfg.max_audio_frame_difference,
                ),
            )

        # shuffle batches
        dp = Shuffler(dp, buffer_size=self.cfg.buffer_batches)

        return dp, shard_list

    def wrap_pipe(self, dp: IterDataPipe, num_shards: int) -> DataLoader:
        assert num_shards > 0
        return DataLoader(
            dp,
            batch_size=None,
            num_workers=1
            if self.cfg.run_debug_mode
            else min(self.cfg.num_workers, num_shards),
            pin_memory=self.cfg.pin_memory,
            persistent_workers=True,
        )


def _filter_on_audio_frame_diff(batch: "DataBatch", threshold: int):
    return batch.audio_frame_difference() < threshold


########################################################################################
# utility functions for initial loading of elements in a pipeline


def decode_wav(value: StreamWrapper):
    assert isinstance(value, StreamWrapper)

    # TODO: switch to new `load` API from torchaudio version 2.1 onwards
    value, sample_rate = torchaudio.load(value)
    assert sample_rate == 16_000

    return value


def decode_json(value: StreamWrapper):
    assert isinstance(value, StreamWrapper)

    return json.load(value)


def decode(element: Tuple[str, StreamWrapper]):
    assert isinstance(element, tuple) and len(element) == 2
    key, value = element

    assert isinstance(key, str)
    assert isinstance(value, StreamWrapper)

    if key.endswith(".wav"):
        value = decode_wav(value)

    if key.endswith(".json"):
        value = decode_json(value)

    return key, value


########################################################################################
# container for a single data sample


@dataclass
class DataSample:
    # identifier of sample
    key: str

    # tensor representing the (input) audio.
    # shape is [1, num_frames]
    audio_tensor: torch.Tensor

    # sampling rate of the (original) audio
    sample_rate: int

    # the amount of frames this audio sample has
    audio_length_frames: int

    # speaker ID
    speaker_id: Optional[str]

    # transcription
    transcription: Optional[str]

    # gender
    gender: Optional[str]

    # language tag
    language_tag: Optional[str]

    def __len__(self):
        return self.audio_length_frames

    def __gt__(self, other):
        return len(self) > len(other)


def construct_data_sample(element: Dict):
    assert isinstance(element, dict)

    json = element[".json"]
    wav = element[".wav"]

    def wrap_to_none(k):
        if k in json:
            return json[k]
        else:
            return None

    ds = DataSample(
        key=json["sample_id"],
        audio_tensor=wav,
        sample_rate=json["sample_rate"],
        audio_length_frames=json["num_frames"],
        speaker_id=wrap_to_none("speaker_id"),
        transcription=wrap_to_none("transcription"),
        gender=wrap_to_none("gender"),
        language_tag=wrap_to_none("language_tag"),
    )

    return ds


########################################################################################
# utility for cropping


def crop_data_sample(x: DataSample, crop_size: int):
    num_frames = x.audio_tensor.shape[1]

    if num_frames <= crop_size:
        return x

    start_idx = torch.randint(low=0, high=num_frames - crop_size, size=())
    end_idx = start_idx + crop_size

    x.audio_tensor = x.audio_tensor[:, start_idx:end_idx]
    x.audio_length_frames = x.audio_tensor.shape[1]

    return x


########################################################################################
# loading data from shards


def find_shards(
    dirs_or_files: Union[str, List[str]],
    shard_file_pattern: str,
    allow_partial: bool,
) -> List[str]:
    if isinstance(dirs_or_files, str):
        dirs_or_files = [dirs_or_files]

    tar_files = []

    for dir_or_file in dirs_or_files:
        dir_or_file = pathlib.Path(dir_or_file)
        if dir_or_file.is_dir():
            tar_files.extend(dir_or_file.rglob(shard_file_pattern))
        if dir_or_file.is_file():
            if fnmatch.fnmatch(dir_or_file.name, shard_file_pattern):
                tar_files.append(dir_or_file)

    # remove potential duplicates by casting to set
    tar_files = list(
        set([str(f) for f in tar_files if allow_partial or ".partial" not in f.name])
    )

    return sorted(tar_files)


def load_samples_from_shards(
    dirs_or_files: Union[str, List[str]],
    shard_file_pattern: str = "*.*.tar*",
    tar_read_mode: str = "r",
    allow_partial: bool = False,
    sample_buffer: int = 100,
    create_infinite_datapipe: bool = False,
    shard_workers: bool = True,
    debug_mode: bool = False,
) -> Tuple[IterDataPipe[DataSample], List[str]]:
    shard_list = find_shards(dirs_or_files, shard_file_pattern, allow_partial)

    if debug_mode:
        shard_list = shard_list[-1:]

    if len(shard_list) <= 0:
        if isinstance(dirs_or_files, list):
            raise ValueError(
                f"unable to find shards in {[str(t) for t in dirs_or_files]}"
            )
        else:
            raise ValueError(f"unable to find shards in {dirs_or_files}")

    # stream of strings representing each shard
    dp = FileLister(shard_list)

    # optionally, make an infinite dataloader
    if create_infinite_datapipe and not debug_mode:
        dp = Cycler(dp)

    # shuffle the stream so order of shards differ
    dp = Shuffler(dp, buffer_size=len(shard_list))

    # each worker only sees 1/n elements
    if shard_workers:
        dp = ShardingFilter(dp)

    # map strings of paths to file handles
    dp = FileOpener(dp, mode="b")

    # expand each file handle to a stream of all files in the tar
    dp = TarArchiveLoader(dp, mode=tar_read_mode)

    # decode each file in the tar to the expected python dataformat
    dp = Mapper(dp, decode)

    # each file in the tar is expected to have the format `{key}.{ext}
    # this groups all files with the same key into one dictionary
    dp = WebDataset(dp)

    # map the dictionary of files with same key to `DataSample` dataclass
    dp = Mapper(dp, construct_data_sample)

    # create a buffer so that batches vary across epochs
    if debug_mode:
        dp = Header(dp, 1)
        if create_infinite_datapipe:
            dp = Cycler(dp)
    else:
        dp = Shuffler(dp, buffer_size=sample_buffer)

    return dp, shard_list


########################################################################################
# Creating a batch of data


@dataclass()
class DataBatch:
    # keys of each sample
    keys: List[str]

    # explicitly store batch size
    batch_size: int

    # audio tensors of each sample (in wav format)
    audio_tensor: torch.Tensor  # shape [BATCH_SIZE, MAX_NUM_FRAMES]

    # store actual num of frames of each sample (padding excluded)
    audio_num_frames: List[int]

    # ground truth values of letters in transcription as integer values
    transcriptions: Optional[List[str]]

    # ground truth values of letters in transcription as integer values
    transcriptions_tensor: Optional[torch.Tensor]  # shape [BATCH_SIZE, MAX_LENGTH]

    # store actual number of letters in transcription (padding excluded)
    transcriptions_length: Optional[List[int]]

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.audio_tensor = self.audio_tensor.pin_memory()
        if self.transcriptions_tensor is not None:
            self.transcriptions_tensor = self.transcriptions_tensor.pin_memory()
        return self

    # move to gpu
    def to(self, device: str):
        self.audio_tensor = self.audio_tensor.to(device, non_blocking=True)
        if self.transcriptions_tensor is not None:
            self.transcriptions_tensor = self.transcriptions_tensor.to(
                device, non_blocking=True
            )

        return self

    def __post_init__(self):
        assert self.batch_size == len(self.keys) == len(self.audio_tensor)
        assert len(self.audio_tensor.shape) == 2
        assert self.audio_tensor.shape[0] == self.batch_size
        assert max(self.audio_num_frames) == self.audio_tensor.shape[1]

        if self.transcriptions_tensor is None:
            return

        assert len(self.transcriptions_length) == self.batch_size
        assert len(self.transcriptions_tensor.shape) == 2
        assert self.transcriptions_tensor.shape[0] == self.batch_size
        assert max(self.transcriptions_length) == self.transcriptions_tensor.shape[1]

    @classmethod
    def from_sample_list(cls, samples: List[DataSample], char_to_idx: Dict[str, int]):
        batch_size = len(samples)

        keys = []
        audio_tensors = []
        audio_num_frames = []
        transcription_strings = []
        transcription_tensors = []
        transcription_lengths = []

        for s in samples:
            keys.append(s.key)
            audio_tensors.append(s.audio_tensor)
            audio_num_frames.append(s.audio_length_frames)

            if char_to_idx is not None:
                transcription_tensor = encode_transcription(
                    s.transcription, char_to_idx
                )

                transcription_strings.append(s.transcription)
                transcription_tensors.append(transcription_tensor)
                transcription_lengths.append(len(s.transcription))

        audio_tensor = collate_append_constant(audio_tensors, variable_dim=1, value=0.0)
        audio_tensor = torch.squeeze(audio_tensor, dim=1)

        if len(audio_tensor.shape) == 1:
            audio_tensor = audio_tensor[None, :]

        if char_to_idx is not None:
            transcription_tensor = collate_append_constant(
                transcription_tensors, variable_dim=0
            )
        else:
            transcription_tensor = None
            transcription_strings = None
            transcription_lengths = None

        return DataBatch(
            keys=keys,
            batch_size=batch_size,
            audio_tensor=audio_tensor,
            audio_num_frames=audio_num_frames,
            transcriptions=transcription_strings,
            transcriptions_tensor=transcription_tensor,
            transcriptions_length=transcription_lengths,
        )

    def audio_frame_difference(self):
        return abs(max(self.audio_num_frames) - min(self.audio_num_frames))


def _map_fn(element: List[DataSample], char_to_idx: Optional[dict]):
    assert isinstance(element, list)
    assert len(element) >= 1
    assert all([isinstance(x, DataSample) for x in element])

    return DataBatch.from_sample_list(element, char_to_idx)


def map_to_batch(
    dp: IterDataPipe[DataSample],
    char_to_idx: Optional[Dict[str, int]] = None,
    max_tokens: Optional[int] = None,
    max_len: Optional[int] = None,
    batch_size: Optional[int] = None,
    drop_last: Optional[bool] = None,  # must be given if batch_size is given
    buffer_size: int = 8,
):
    if max_tokens is None and batch_size is None:
        raise ValueError(f"one of {max_tokens=} or {batch_size=} must be given")
    if max_tokens is not None and batch_size is not None:
        raise ValueError(f"one of {max_tokens=} or {batch_size=} must be given")

    map_fn = functools.partial(_map_fn, char_to_idx=char_to_idx)

    if max_tokens is not None:
        dp = MaxTokenBucketizer(
            dp,
            max_token_count=max_tokens,
            max_len=max_len,
            include_padding=True,
            buffer_size=buffer_size,
        )

        return Mapper(dp, fn=map_fn)
    elif batch_size is not None:
        if drop_last is None:
            raise ValueError(f"{drop_last=} must be defined if {batch_size=}")
        return Batcher(
            dp, batch_size=batch_size, drop_last=drop_last, wrapper_class=map_fn
        )
