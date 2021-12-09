"""
Sets up the project as a package. Allows us to pip install it and use imports even when not in PyCharm.

To activate it all you need is to activate your virtual environment run the following once:
'pip install -e .'
... inside the project root dir. No more need to modify PYTHONPATH baby.

NB! An __init__.py file is required inside every dir that is to be included.
"""
from setuptools import setup, find_packages

# Package configuration
setup(
    name='mehra-prescott',
    version='0.0.1',
    description='',
    long_description=open("README.md", 'r').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
)