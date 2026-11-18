from pathlib import Path

from setuptools import setup, find_packages

README = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

setup(
    name="gossamer-threaded-intelligence",  # Distribution name (unique in index)
    # Artifact Registry REFUSES to overwrite an existing version, so a forgotten
    # bump here silently republishes nothing and ships a stale wheel. Keep in step
    # with maneuver-map/scripts/versions.env (check_versions.sh asserts it).
    version="0.13.2",
    description="Gossamer Threaded Intelligence: swarm coordination algorithms + swappable DCC primitives, a unified coordination-task/quality API, peer-state predictors, graph-based message-passing policies, information-theoretic + criticality metrics, a MARL toolkit, and the Arboria Swarm Benchmark.",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Arboria Labs",
    license="MIT",
    license_files=["LICENSE"],
    url="https://www.arborialabs.com",
    project_urls={
        "Source": "https://github.com/silicondaydream/gossamer-threaded-intelligence",
        "Research": "https://www.arborialabs.com/research",
    },
    packages=find_packages(include=["gossamer", "gossamer.*"]),  # Import remains `gossamer`
    install_requires=[
        "numpy",
        "networkx",
        "pandas",
        "scipy",
        "scikit-learn",
    ],
    # These were installed ad-hoc in CI rather than declared, so a fresh checkout
    # could not run the MAPPO baselines or the MILP comparator at all.
    extras_require={
        "learning": ["torch"],      # gossamer.learning.mappo, the MAPPO baselines
        "milp": ["ortools"],        # the HMA central-planner comparator
        "bench": ["tabulate"],      # leaderboard rendering
        "all": ["torch", "ortools", "tabulate"],
    },
    entry_points={
        "console_scripts": [
            # The benchmark is meant to be publicly runnable (the N2 moat); make it
            # runnable without knowing the module path.
            "gossamer-bench=gossamer.benchmarks.cli:main",
        ],
    },
    # The versions CI tests (3.11-3.13).
    python_requires=">=3.11",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
    ],
)
