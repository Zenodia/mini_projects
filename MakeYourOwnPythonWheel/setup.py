from setuptools import find_packages, setup
from package import Package

setup(
    name="zeno-package",
    author="Zenodia Charpy",
    author_email="zecharpy@microsoft.com",
    url="localhost",
    packages=find_packages(),
    include_package_data=True,
    cmdclass={
        "package": Package
    }
)
