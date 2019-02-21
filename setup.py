from setuptools import setup, find_packages
import subprocess

result = subprocess.run(["git", "describe", "--abbrev=0"], stdout=subprocess.PIPE)
VERSION = result.stdout.decode("utf-8").rstrip()

with open("README.md") as file:
    long_description = file.read()

install_requires = [
    "numpy>=1.14.3",
    "scipy>=1.1.0",
    "Pillow>=5.1.0",
    "tqdm==4.28.1",
    "numba==0.42.0",
]

setup(
    name="gorl",
    version=VERSION,
    url="https://github.com/denistr16/GoL-RL/",
    zip_safe=False,
    packages=find_packages(),
    include_package_data=True,
    description="Playground for reinforcement learning with John Conway Game of Life",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="GoL-RL team",
    license="MIT",
    install_requires=install_requires,
)
