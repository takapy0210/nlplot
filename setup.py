import os, sys
from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE.txt') as f:
    license = f.read()


def read_requirements():
    """Parse requirements from requirements.txt."""
    reqs_path = os.path.join('.', 'requirements.txt')
    with open(reqs_path, 'r') as f:
        requirements = [line.rstrip() for line in f]
    return requirements


setup(
    name='nlplot',
    version='1.0.0',
    description='Visualization Module for Natural Language Processing',
    long_description=readme,
    long_description_content_type='text/markdowm',
    author='Takanobu Nozawa',
    author_email='takanobu.030210@gmail.com',
    url='https://github.com/takapy0210/nlplot',
    license=license,
    install_requires=read_requirements(),
    packages=find_packages(exclude=('tests')),
    package_data={'nlplot':['data/*']},
    python_requires='~=3.6',
    classifiers=[
        'License::OSIApproved::MITLicense',
        'ProgrammingLanguage::Python::3',
        'ProgrammingLanguage::Python::3.6',
        'ProgrammingLanguage::Python::3.7',
        'ProgrammingLanguage::Python::3.8',
        'OperatingSystem::OSIndependent',
    ],
)