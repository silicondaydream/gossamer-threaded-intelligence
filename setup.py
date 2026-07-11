from setuptools import setup, find_packages

setup(
    name="gossamer-threaded-intelligence",  # Distribution name (unique in index)
    # Artifact Registry REFUSES to overwrite an existing version, so a forgotten
    # bump here silently republishes nothing and ships a stale wheel. Keep in step
    # with maneuver-map/scripts/versions.env (check_versions.sh asserts it).
    version="0.7.0",
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
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
