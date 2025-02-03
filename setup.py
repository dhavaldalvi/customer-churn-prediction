from setuptools import find_packages, setup
from typing import List

DASH_DOT_E = '-e .'
def get_requirements(file_path:str)->List[str]:
    '''
    This function will returns list of required libraries and packages.
    '''
    list_of_requirements = []
    with open(file_path) as file:
        list_of_requirements = file.readlines()
        list_of_requirements = [requirements.replace('\n','') for requirements in list_of_requirements]

        if DASH_DOT_E in list_of_requirements:
            list_of_requirements.remove(DASH_DOT_E)

    return list_of_requirements



setup(
    name='customer-churn-prediction',
    version='0.0.1',
    author='Dhaval',
    author_email='dhavaldalvi9@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)