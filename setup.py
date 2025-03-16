from setuptools import setup, find_packages

setup(
    name="computeruse",
    version="0.1.0",
    description="LLM-powered Windows desktop automation",
    author="Shrey",
    packages=find_packages(),
    install_requires=[
        "python-dotenv>=1.0.0",
        "pydantic>=2.0.0",
        "langchain-core>=0.1.0",
        "asyncio>=3.4.3",
        "pillow>=10.0.0",
    ],
    extras_require={
        "openai": ["langchain-openai>=0.0.5", "openai>=1.0.0"],
        "anthropic": ["langchain-anthropic>=0.1.0"],
        "windows": ["uiautomation>=2.0.0", "pywinauto>=0.6.8", "comtypes>=1.1.14"],
    },
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
)