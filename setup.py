from setuptools import setup, find_packages

setup(
  name = 'RIN-pytorch',
  packages = find_packages(exclude=[]),
  version = '0.0.1',
  license='MIT',
  description = 'RIN - Recurrent Interface Network - Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/lucidrains/RIN-pytorch',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'attention mechanism',
    'denoising diffusion',
    'image and video generation'
  ],
  install_requires=[
    'einops>=0.6',
    'torch>=1.6',
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
