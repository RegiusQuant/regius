# -*- coding: utf-8 -*-

from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='regius-dl',
    version='0.0.1',
    packages=['regius'],
    description='Regius Deep Learning Toolbox',
    long_description=long_description,
    author='Jiang Yize',
    url='https://github.com/RegiusQuant/regius',
    author_email='315135833@qq.com',
    license='Apache License 2.0',
    keywords=['deeplearning']
)

