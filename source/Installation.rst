Installation
====================

We recommend the requirements as follows:

1.	Clone this repository and navigate to the folder

    .. code-block:: bash

       git clone https://github.com/DLCV-BUAA/TinyLLaVABench.git

       cd TinyLLaVABench


2.	Install Package

    .. code-block:: bash

       conda create -n tinyllava_factory python=3.10 -y

       conda activate tinyllava_factory

       pip install --upgrade pip  # enable PEP 660 support

       pip install -e .


3.	Install additional packages for training cases

    .. code-block:: bash

       pip install flash-attn --no-build-isolation
