from os import path
from setuptools import find_packages, setup

here = path.abspath(path.dirname(__file__))

setup(
    name='odin',
    version='0.1',
    description='A Theano framework for building and training neural networks',
    url='https://github.com/mila-udem/blocks',
    author='University of Montreal',
    license='MIT',
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Speech Processing',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.5',
    ],
    keywords='tensorflow pytorch machine learning neural networks deep learning bayesian',
    packages=find_packages(exclude=['examples', 'docs', 'doctests', 'tests']),
    scripts=['bin/speech-augmentation', 'bin/speech-test'],
    setup_requires=['numpy'],
    install_requires=['numpy', 'six', 'pyyaml', 'toolz', 'theano',
                      'picklable-itertools', 'progressbar2', 'fuel'],
    extras_require={
        'test': ['mock', 'nose', 'nose2'],
        'docs': ['sphinx', 'sphinx-rtd-theme']
    },
    zip_safe=False)
