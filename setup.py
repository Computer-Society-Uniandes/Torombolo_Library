from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='torombolo',
    version='0.1',
    packages=find_packages(),
    install_requires=requirements,
    author='AI Group Uniandes',
    description='Machine Learning AI library for Python',
)
