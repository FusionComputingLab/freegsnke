import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="freegsnke",
    version="0.3.0",
    description="An extension of FreeGSFast with a Newton-Krylov Grad-Shafranov solver and evolutive capabilities.",
    long_description=long_description,
    url="https://github.com/farscape-project/freegsnke",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)",
        "Operating System :: OS Independent",
    ],
    packages=["freegsnke"],
    python_requires=">=3.0"
)