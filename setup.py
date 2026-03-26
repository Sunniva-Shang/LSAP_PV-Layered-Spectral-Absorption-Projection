from setuptools import setup

setup(
    name="improved-diffusion",
    py_modules=["improved_diffusion"],
    install_requires=["blobfile>=0.11.0", "torch", "tqdm"],
)
