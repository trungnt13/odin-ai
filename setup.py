from os import path
from setuptools import find_packages, setup

here = path.abspath(path.dirname(__file__))

long_description = \
'''
An end-to-end framework support multi-modal data processing
and fast prototyping of machine learning algorithm in form
of organized networks.
'''

setup(
    name='odin',
    version='0.1',
    description="Deep learning for research and production",
    long_description=long_description,
    url='https://github.com/imito/odin',
    author='Trung Ngo Trong',
    author_email='trung@imito.ai',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Speech Processing',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
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
    keywords='tensorflow pytorch machine learning neural networks deep learning bayesian',
    packages=find_packages(exclude=['examples', 'docs', 'tests']),
    # scripts=['bin/speech-augmentation', 'bin/speech-test'],
    install_requires=['numpy>=1.9.1',
                      'scipy>=0.14',
                      'six>=1.9.0',
                      'scikit-learn>=0.20.0',
                      'tqdm',
                      'dill',
                      'pycrypto'],
    extras_require={
        'visualize': ['pydot>=1.2.4',
                      'colorama',
                      'seaborn',
                      'matplotlib>=3.0.0'],
        'tests': ['pytest',
                  'pandas',
                  'requests'],
        'audio': ['soundfile',
                  'resampy'],
        'docs': ['sphinx', 'sphinx-rtd-theme']
    },
    zip_safe=False)
