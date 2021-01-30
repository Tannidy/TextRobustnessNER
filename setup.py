#!/usr/bin/env python
# coding=utf-8
from setuptools import setup, find_packages

with open('README.md', encoding='utf-8') as f:
    readme = f.read()

with open('LICENSE', encoding='utf-8') as f:
    license = f.read()

with open('requirements.txt', encoding='utf-8') as f:
    reqs = f.read()

pkgs = [p for p in find_package() if p.startswith('TextRobustness')]
print(pkgs)

setup(
    name='TextRobustness',
    version='0.0.1',
    url='',
    description='',
    long_description=readme,
    long_description_content_type='text/markdown',
    license='Apache License',
    author='Fudan NLP Team',
    python_requires='>=3.6',
    packages=pkgs,
    install_requires=reqs.strip().split('\n'),
)
