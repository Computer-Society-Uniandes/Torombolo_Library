from setuptools import setup, find_packages

VERSION = '0.0.1'

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='torombolo',
    version=VERSION,
    packages=find_packages(),
    install_requires=requirements,
    author='AI Group Uniandes',
    description='Machine Learning AI library for Python',
    long_description=open('README.md').read(),
    url='https://github.com/Computer-Society-Uniandes/Torombolo_Library',
    license='MIT',
    python_requires='>=3.6'
)