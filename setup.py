from os import path

from setuptools import find_packages, setup

ODIN_VERSION = '1.4.0'

TENSORFLOW_VERSION = '2.5.0'
TFP_VERSION = '0.13.0'
PYTORCH_VERSION = '1.9.0+cu111'
PYTORCH_VISION = '0.10.0+cu111'
PYTORCH_AUDIO = '0.9.0'

# ===========================================================================
# Dependencies
# ===========================================================================
dependencies = [
  'numpy',
  'scipy',
  f"tensorflow=={TENSORFLOW_VERSION}",
  f'tensorflow-probability=={TFP_VERSION}',
  # f'torch=={PYTORCH_VERSION}',
  # f'torchvision=={PYTORCH_VISION}',
  # f'torchaudio=={PYTORCH_AUDIO}',
  # 'pyro-ppl',
  'tensorflow-addons',
  'tensorflow-datasets',
  'transformers',
  'hydra-core>=1.0.0',
  'bigarray>=0.2.1',
  'six',
  'scikit-learn',
  'matplotlib',
  'decorator',
  'tqdm',
  'pyyaml',
  'pycrypto',
  'typeguard'  # runtime type check
]
# ===========================================================================
# Description
# ===========================================================================
here = path.abspath(path.dirname(__file__))

long_description = \
  '''
  An end-to-end framework support multi-modal data processing
  and fast prototyping of machine learning algorithm in form
  of organized networks.
  '''

# ===========================================================================
# Setup
# ===========================================================================
setup(
  name='odin-ai',
  version=ODIN_VERSION,
  description="Deep learning for research and production",
  long_description=long_description,
  long_description_content_type='text/x-rst',
  url='https://github.com/imito/odin-ai',
  author='Trung Ngo Trong',
  author_email='trungnt13@gmail.com',
  license='MIT',
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Intended Audience :: Education',
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Topic :: Scientific/Engineering :: Information Analysis',
    'Topic :: Scientific/Engineering :: Bio-Informatics',
    'Topic :: Multimedia :: Sound/Audio :: Speech',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.7',
    'Natural Language :: English',
    'Operating System :: MacOS :: MacOS X',
    'Operating System :: Microsoft :: Windows',
    'Operating System :: POSIX :: Linux',
  ],
  keywords=
  'tensorflow pytorch machine learning neural networks deep learning bayesian',
  packages=find_packages(exclude=['examples', 'examples/*', 'docs', 'tests']),
  # scripts=['bin/speech-augmentation', 'bin/speech-test'],
  setup_requires=['pip>=19.0'],
  install_requires=dependencies,
  extras_require={
    'visualize': ['pydot>=1.2.4', 'colorama', 'seaborn'],
    'tests': ['pytest', 'pandas', 'requests'],
    'audio': ['soundfile', 'resampy'],
    'docs': ['sphinx', 'sphinx_rtd_theme']
  },
  zip_safe=False)
