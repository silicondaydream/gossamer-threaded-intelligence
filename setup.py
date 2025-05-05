from setuptools import setup, find_packages

setup(
    name="gossamer_intelligence",
    version="0.1.0",
    description="Gossamer Threaded Intelligence: decentralized agent algorithms for swarm intelligence",
    author="Arboria Research",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "networkx",
        "pandas",
        "scipy",
        "scikit-learn",
    ],
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)