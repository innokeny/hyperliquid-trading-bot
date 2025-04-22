from setuptools import setup, find_packages

setup(
    name="hyperliquid-trading-bot",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "hyperliquid-python-sdk>=0.1.18",
        "torch>=2.0.0",
        "catboost>=1.2.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "websockets>=11.0.3",
        "ta>=0.10.0",
        "python-dotenv>=1.0.0",
        "pydantic>=2.0.0",
        "loguru>=0.7.0",
        "structlog>=23.1.0",
    ],
    python_requires=">=3.8",
) 