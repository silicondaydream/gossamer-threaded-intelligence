from setuptools import setup, find_packages

setup(
    name="gossamer-threaded-intelligence",  # Distribution name (unique in index)
    version="0.5.0",
    description="Gossamer Threaded Intelligence: swarm coordination algorithms + swappable DCC primitives, a unified coordination-task/quality API, peer-state predictors, graph-based message-passing policies, information-theoretic + criticality metrics, a MARL toolkit, and the Arboria Swarm Benchmark.",
    author="Arboria Labs",
    license="MIT",
    packages=find_packages(include=["gossamer", "gossamer.*"]),  # Import remains `gossamer`
    install_requires=[
        "numpy",
        "networkx",
        "pandas",
        "scipy",
        "scikit-learn",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
