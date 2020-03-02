from os import path

from setuptools import find_packages, setup

_ODIN_VERSION_ = '1.2.3'

_TENSORFLOW_VERSION = '2.1.0'
_TENSORFLOW_PROBABILITY_VERSION = '0.9.0'
_TENSORFLOW_ADDONS_VERSION = '0.5.2'
_PYTORCH_VERSION = '1.4.0'


# ===========================================================================
# Packages check
# ===========================================================================
def get_tensorflow_version():
  import subprocess
  try:
    task = subprocess.Popen(["nvcc", "--version"],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    out = task.stdout.read()
    if "Cuda" in str(out, 'utf-8'):
      return "tensorflow==%s" % _TENSORFLOW_VERSION
  except FileNotFoundError:
    pass
  return "tensorflow-cpu==%s" % _TENSORFLOW_VERSION


# ===========================================================================
# Dependencies
# ===========================================================================
dependencies = [
    'numpy>=1.0.0',
    get_tensorflow_version(),
    'tensorflow-probability==%s' % _TENSORFLOW_PROBABILITY_VERSION,
    'tensorflow-addons==%s' % _TENSORFLOW_ADDONS_VERSION,
    'tensorflow-datasets',
    'torch==%s' % _PYTORCH_VERSION,
    'hydra-core', # for easy configuration
    'pytorch-lightning',  # for training pytorch module
    'bigarray>=0.2.1',
    'six>=1.9.0',
    'scikit-learn==0.22.1',
    'matplotlib>=3.0.0',
    'decorator',
    'tqdm',
    'dill',
    'pyyaml',
    'pycrypto',
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
    version=_ODIN_VERSION_,
    description="Deep learning for research and production",
    long_description=long_description,
    long_description_content_type='text/x-rst',
    url='https://github.com/imito/odin-ai',
    author='Trung Ngo Trong',
    author_email='trung@imito.ai',
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
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
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
