import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
    name='motionmapperpy',
    version='1.0',
    author="Kanishk Jain",
    author_email="kanishkbjain@gmail.com",
    maintainer="Kanishk Jain",
    maintainer_email="kanishkbjain@gmail.com",
    description="A modified Python Implemention of MotionMapper (https://github.com/gordonberman/MotionMapper)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bermanlabemory/motionmapperpy",
    download_url="https://github.com/bermanlabemory/motionmapperpy.git",
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "h5py",
        "hdf5storage",
        "scikit-learn",
        "scikit-image",
        "easydict",
        "umap-learn"
    ],
    packages=setuptools.find_packages(),
    classifiers=[
    "Programming Language :: Python :: 3",
    "Intended Audience :: Science/Research"
    "Operating System :: OS Independent",
    "Development Status :: 4 - Beta",
    ""
    ],
    )