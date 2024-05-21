from distutils.core import setup

from setuptools import find_packages

setup(
    name="nanow2v2",
    version="1.0",
    description="Self-contained implementation of wav2vec2",
    author="anon",
    author_email="anon@gmail.com",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "convert_to_wav=nanow2v2.scripts.convert_to_wav:main",
            "verify_checksum=nanow2v2.scripts.verify_checksum:main",
            "write_tar_shards=nanow2v2.scripts.write_tar_shards:main",
            "split_csv=nanow2v2.scripts.split_csv:main",
        ],
    },
)
