import os
from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()


def read_requirements():
    """Parse requirements from requirements.txt."""
    reqs_path = os.path.join('.', 'requirements.txt')
    with open(reqs_path, 'r') as f:
        requirements = [line.rstrip() for line in f]
    return requirements


setup(
    name='nlplot',
    version='1.3.0',
    description='Visualization Module for Natural Language Processing',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Takanobu Nozawa',
    author_email='takanobu.030210@gmail.com',
    url='https://github.com/takapy0210/nlplot',
    license='MIT License',
    install_requires=read_requirements(),
    packages=find_packages(exclude=('tests')),
    tests_require=["pytest"],
    package_data={'nlplot': ['data/*']},
    python_requires='>=3.6'
)