"""Install LL3M."""

import os
import sys
import setuptools

# Get the long description from the README file.
with open('README.md') as fp:
  _LONG_DESCRIPTION = fp.read()

_jax_version = '0.4.23'
_jaxlib_version = '0.4.23'

setuptools.setup(
    name='LL3M',
    version='0.0.1',
    description='cockatoo in jax',
    long_description=_LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    author='Jiasen Lu',
    author_email='jiasenl@allenai.org',
    url='https://github.com/jiasenlu/LL3M',
    license='Apache 2.0',
    packages=setuptools.find_packages(),
    package_data={
        '': ['**/*.gin'],  # not all subdirectories may have __init__.py.
    },
    scripts=[],
    install_requires=[
        # 'airio @ git+https://github.com/google/airio#egg=airio',
        'airio @ git+https://github.com/google/airio.git@1a5259c3c3794557433c2094c9c8aa76cea69000',
        'absl-py',
        'cached_property',
        'clu @ git+https://github.com/google/CommonLoopUtils#egg=clu',
        'flax @ git+https://github.com/google/flax#egg=flax',
        'fiddle >= 0.2.5',
        'gin-config',
        f'jax == {_jax_version}',
        f'jaxlib == {_jaxlib_version}',
        (
            'jestimator @'
            ' git+https://github.com/google-research/jestimator#egg=jestimator'
        ),
        'numpy',
        'optax @ git+https://github.com/deepmind/optax#egg=optax',
        'orbax-checkpoint',
        'seqio @ git+https://github.com/google/seqio#egg=seqio',
        'tensorflow-cpu',
        'tensorstore >= 0.1.20',
        # remove this when sentencepiece_model_pb2 is re-generated in the
        # sentencepiece package.
        'protobuf==3.20.3',
        'lightning_utilities',
        'jsonargparse',
        'lightning_utilities',
        'einops',
        'wandb',
        'transformers==4.31.0',
        'datasets==2.14.2',
        'huggingface_hub==0.16.4',
        'tqdm',
        'h5py',
        'ml_collections',
        'requests',
        'mlxu @ git+https://github.com/jiasenlu/mlxu#egg=mlxu',
        'sentencepiece',
        'gcsfs==2023.9.2',
        'scipy==1.12.0',
    ],
    extras_require={
        'gcp': [
            'gevent',
            'google-api-python-client',
            'google-compute-engine',
            'google-cloud-storage',
            'oauth2client',
        ],
        'mm_eval': [
            'pycocoevalcap',
            'nltk'
        ],
        'test': ['pytest', 't5'],
        # Cloud TPU requirements.
        'tpu': [f'jax[tpu] == {_jax_version}'],
        'gpu': [
            'ipdb==0.13.9',
            'fasttext==0.9.2',
            'pysimdjson==5.0.2',
            'pytablewriter==0.64.2',
            'gdown==4.5.3',
            'best-download==0.0.9',
            'lm_dataformat==0.0.20',
            'dllogger@git+https://github.com/NVIDIA/dllogger#egg=dllogger',
            'tfds-nightly',
            't5==0.9.4',
        ],
        'serve': [
            'pydantic',
            'uvicorn',
            'gradio',
            'fastapi',
        ],
        'eval':[
            'lm_eval==0.3.0',
        ]
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='multimodal text nlp machinelearning',
)