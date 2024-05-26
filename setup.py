from setuptools import setup

setup(
    name="yolo-mp-object-tracker",
    version="0.1",
    py_modules=["object_tracker", "tracked_object", "yolo_object", "hand_helper"],
    install_requires=[
        "numpy"
    ],
    url="https://github.com/grgzpp/yolo-mp-object-tracker",
    author="Giorgio Zoppi",
    description="Object tracker that provides functionality to track objects detected by the YOLO object detection system, combined with information provided by the MediaPipe hand tracker.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)