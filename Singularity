Bootstrap: docker
From: tensorflow/tensorflow:1.12.3-gpu-py3

%files
    # Copy important files here (code and model files)
    requirements.txt requirements.txt
    segmentation_model/ mnms/

%post
    apt-get -y update
    apt-get install -y libsm6 libxext6 libxrender-dev
    pip install -r requirements.txt

%runscript
    echo "tensorflow container"
    echo

    python /mnms/segmentation_model/eval.py "$@"

%labels
    Maintainer "Victor M. Campello"