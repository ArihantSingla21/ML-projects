from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = "-e ."

def get_requirements(file_path: str) -> List[str]:
    """Reads requirements from a file and returns a list of strings."""
    with open(file_path, 'r') as file_obj:
        requirements = [line.strip() for line in file_obj.readlines() if line.strip()]
    
    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)
    
    return requirements

setup(
    name='mlproject',
    version='0.0.1',  # Version should be a string
    author='arihant',
    author_email='arihantsingla21@gmail.com',
    packages=find_packages(),  # Finds all packages under the current directory
    install_requires=get_requirements('requirements.txt')
)
