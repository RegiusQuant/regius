# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

exec(open('regius/version.py').read())

setup(
    name='regius-dl',
    version=__version__,
    packages=find_packages(),

    description='Regius Deep Learning Toolbox',
    long_description=readme,
    long_description_content_type='text/markdown',

    author='Jiang Yize',
    author_email='315135833@qq.com',
    url='https://github.com/RegiusQuant/regius',

    license='Apache Software License 2.0',
    keywords=['deep learning', 'machine learning'],

    classifiers=[
        'Development Status :: 1 - Planning',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: Chinese (Simplified)',
        'Programming Language :: Python :: 3.7',
    ],

    install_requires=[
        'pandas',
        'torch',
    ]
)
