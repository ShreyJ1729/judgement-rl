from setuptools import setup, find_packages
import os


# Read README for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as fh:
            return fh.read()
    return ""


# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if os.path.exists(requirements_path):
        with open(requirements_path, "r", encoding="utf-8") as fh:
            return [
                line.strip() for line in fh if line.strip() and not line.startswith("#")
            ]
    return []


setup(
    name="judgement-rl",
    version="2.0.0",
    author="Judgement RL Team",
    author_email="team@judgement-rl.com",
    description="Reinforcement Learning for the Judgement Card Game",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/judgement-rl",
    project_urls={
        "Bug Reports": "https://github.com/your-repo/judgement-rl/issues",
        "Source": "https://github.com/your-repo/judgement-rl",
        "Documentation": "https://github.com/your-repo/judgement-rl/docs",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Games/Entertainment :: Board Games",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-mock>=3.8.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
            "pre-commit>=2.20.0",
        ],
        "gui": [
            # GUI dependencies will be added when GUI is implemented
        ],
        "monitoring": [
            "tensorboard>=2.10.0",
            "wandb>=0.13.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "judgement-train=judgement_rl.cli.train:main",
            "judgement-evaluate=judgement_rl.cli.evaluate:main",
            "judgement-gui=judgement_rl.cli.gui:main",
            "judgement-monitor=judgement_rl.cli.monitor:main",
        ],
    },
    include_package_data=True,
    package_data={
        "judgement_rl": ["py.typed"],
    },
    zip_safe=False,
    keywords="reinforcement-learning, card-game, ppo, self-play, judgement",
)
