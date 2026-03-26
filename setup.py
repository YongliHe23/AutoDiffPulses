from setuptools import setup, find_packages
import ctypes


def cuda_is_available():
    libnames = ('libcuda.so', 'libcuda.dylib', 'cuda.dll')
    for name in libnames:
        try:
            ctypes.CDLL(name)
        except OSError:
            continue
        else:
            return True
    else:
        return False
    return False


REQUIRED_PACKAGES = ['torch>=1.6', 'numpy', 'scipy']

# REQUIRED_PACKAGES = ['torch>=1.6', 'numpy', 'cupy>=7.0.0', 'mrphy>=0.1.5']
# if not cuda_is_available():
#     REQUIRED_PACKAGES.remove('cupy>=7.0.0')

with open("README.md", "r") as h:
    long_description = h.read()

setup(
    name="adpulses",
    version="0.3.0",
    author="Tianrui Luo, Yongli He",
    author_email="tianrluo@umich.edu, yonglihe@umich.edu",
    description="A Pytorch based tool for MR pulse design, with support for steady-state and SMS sequence optimization",
    license='MIT',
    long_description=long_description,
    long_description_content_type="text/x-rst",
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
)
