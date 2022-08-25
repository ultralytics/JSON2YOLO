from setuptools import setup

setup(name='json2yolo',
      version='0.0.1',
      author='Ultralytics',
      packages=['json2yolo'],
      description='Dataset converter for yolo',
      license='GPL',
      install_requires=[
        "numpy",
        "opencv-python>=4.1.2",
        "pandas",
        "Pillow",
        "pyYAML",
        "requests",
        "tqdm",

      ],
)
