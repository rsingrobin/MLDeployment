from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:

    """
    This function reads the requirements file and returns a list of packages.
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [requirement.replace("\n","") for requirement in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
            
    return requirements

setup(
    name='ML_Deployment',
    version='0.0.1',
    author='Sing',
    author_email='rsingrobin@gmail.com',
    packages=find_packages(),
    #install_requires=['pandas', 'scikit-learn', 'numpy', 'flask', 'gunicorn']
    install_requires=[get_requirements('requirements.txt')]
)