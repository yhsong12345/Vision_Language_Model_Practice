from setuptools import setup, find_packages

with open("requirements.txt", "r") as f:
    requirements = []
    for line in f:
        line = line.strip()
        # Skip comments and empty lines
        if line and not line.startswith("#"):
            requirements.append(line)

try:
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "VLA Training Framework"

setup(
    name="vla",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Vision-Language-Action Model Training Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/vla",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
        "robot": [
            "lerobot>=0.1.0",
        ],
        "flash": [
            "flash-attn>=2.5.0",
        ],
        "mujoco": [
            "gymnasium[mujoco]>=0.29.0",
        ],
    },
)
