from setuptools import setup, find_packages
from typing import List


HYPEN_E_DOT='-e .'
def get_requirements(file_path:str)->List[str]:
    """
    Reads a requirements file and returns a list of package dependencies.

    Args:
        file_path (str): Path to the requirements file.

    Returns:
        List[str]: A list of package names as strings, with any editable 
                   install commands (like '-e .') removed.
                   
    Example:
        If requirements.txt contains:
            numpy>=1.20.0
            pandas>=1.2.0
            -e .
        This function will return:
            ['numpy>=1.20.0', 'pandas>=1.2.0']
    """
    
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements

setup(
    name="insurance-pricing-prediction",
    version="0.1.0",
    author="Olawumi Salaam",
    author_email="olawumisalaam@gmail.com",
    description="A machine learning project to predict health insurance pricing using XGBoost.",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
    python_requires=">=3.8",
)