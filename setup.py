from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='feda_project',
    version='1.0.0',
    author='Tawan Teopipithaporn',
    description='A Forest-guided Estimation of Distribution Algorithm (FEDA) for optimization.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/tortawan/feda-project', # Replace with your repo URL
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    # Core dependencies needed for your project to run
    install_requires=[
        'numpy>=1.21.0',
        'scikit-learn>=1.0.0',
        'matplotlib>=3.0.0',
    ],
    # Extra dependencies for development and testing
    extras_require={
        'test': ['pytest>=7.0.0'],
    },
)