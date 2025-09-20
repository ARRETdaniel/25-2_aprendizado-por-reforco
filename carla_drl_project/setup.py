#!/usr/bin/env python3
"""
CARLA Deep Reinforcement Learning Project Setup
================================================

This package provides a complete framework for training Deep Reinforcement Learning
agents (DDPG, TD3, SAC) in CARLA simulator using ROS 2 for sensor data and control.

Author: [Your Name]
Version: 0.1.0
License: MIT
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="carla-drl-project",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Deep Reinforcement Learning with CARLA 0.9.16 and ROS 2",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/carla-drl-project",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.1.0",
            "pytest-cov>=3.0.0",
            "black>=22.3.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0",
        ],
        "distributed": [
            "ray[tune]>=2.0.0",
            "redis>=4.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "carla-drl-train=scripts.train_agent:main",
            "carla-drl-evaluate=scripts.evaluate_agent:main",
            "carla-drl-record=scripts.record_episodes:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["config/*.yaml", "data/*"],
    },
    zip_safe=False,
)
