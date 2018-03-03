"""Certify Poisoning Attack Defenses"""

from distutils.core import setup
from certml import __version__

setup(
    name='certml',
    version=__version__,
    author=('Christopher Frederickson'),
    author_email=('fredericc0@students.rowan.edu'),
    packages=['certml'],
    url='',
    description='Certify data-based poisoning attack defenses. Based on work by Steinhardt et al. See: https://github.com/chrisfrederickson/data-poisoning-release',
    long_description=open('README.rst').read(),
    install_requires=[  # TODO Update this!
        'numpy',
        'scipy',
        'scikit-learn',
    ],
)