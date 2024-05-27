Evaluation
====================

We currently provide evaluations on 8 benchmarks, including VQAv2, GQA, ScienceQA, ScienceQA, POPE, MME, MM-Vet and MMMU. 

For VQAv2, GQA, ScienceQA, POPE, MME and MM-Vet, you **MUST first download** eval.zip_. It contains custom annotations, scripts, and the prediction files with LLaVA v1.5. Please extract it to ``path/to/your/dataset/eval``.
Or you can just follow the evaluation_ instructions of LLaVA v1.5.

.. _eval.zip: https://drive.google.com/file/d/1atZSBBrAX54yYpxtVVW33zFvcnaHeFPy/view

.. _evaluation: https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md

For MMMU, you **MUST first download** MMMU.zip_. It contains custom annotations and scripts. Please extract it to ``path/to/your/dataset/eval/MMMU``.

.. _MMMU.zip: https://drive.google.com/file/d/1TJszQ23X-7TeMYDA7hVKpoHy9yo-lsc5/view?usp=sharing


VQAv2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1.	**Dataset:** Download test2015_ and put it under ``path/to/your/dataset/eval/vqav2``.

.. _test2015: http://images.cocodataset.org/zips/test2015.zip

2. Please change ``MODEL_PATH``, ``MODEL_NAME``, ``EVAL_DIR``, and ``conv-mode`` in ``scripts/eval/vqav2.sh``.

3.	**Inference:** VQAv2 supports multi-gpus inference with the following command.

   .. code-block:: bash

      cd TinyLLaVA_Factory
      CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/eval/vqav2.sh


4.	Submit the results(``path/to/your/dataset/eval/vqav2/answers_upload``) to the vqav2_evaluation_server_.

.. _vqav2_evaluation_server: https://eval.ai/web/challenges/challenge-page/830/my-submission

GQA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1.	**Dataset:** Download the data_ and evaluation_scripts_ following the official instructions and put under ``path/to/your/dataset/eval/gqa/data``.

.. _data: https://cs.stanford.edu/people/dorarad/gqa/download.html
.. _evaluation_scripts: https://cs.stanford.edu/people/dorarad/gqa/evaluate.html

2. Please change ``MODEL_PATH``, ``MODEL_NAME``, ``EVAL_DIR``, and ``conv-mode`` in ``scripts/eval/gqa.sh``.

3.	**Inference:** GQA supports multi-gpus inference with the following command.

    .. code-block:: bash

       cd TinyLLaVA_Factory
       CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/eval/gqa.sh

ScienceQA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1.	**Dataset:** Under ``path/to/your/dataset/eval/scienceqa``, download ``images``, ``pid_splits.json``, ``problems.json`` from the ``scienceqa`` folder of the ScienceQA repo_.

.. _repo: https://github.com/lupantech/ScienceQA

2. Please change ``MODEL_PATH``, ``MODEL_NAME``, ``EVAL_DIR``, and ``conv-mode`` in ``scripts/eval/sqa.sh``.

3.	**Inference:** ScienceQA does not support multi-gpus inference, please use the following command for single-gpu inference.

   .. code-block:: bash

      cd TinyLLaVA_Factory
      CUDA_VISIBLE_DEVICES=0 bash scripts/eval/sqa.sh

TextVQA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1.	**Dataset:** Download TextVQA_0.5.1_val.json_ and images_ and extract to ``path/to/your/dataset/eval/textvqa``.

.. _TextVQA_0.5.1_val.json: https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json
.. _images: https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip

2. Please change ``MODEL_PATH``, ``MODEL_NAME``, ``EVAL_DIR``, and ``conv-mode`` in ``scripts/eval/textvqa.sh``.

3.	**Inference:** TextVQA does not support multi-gpus inference, please use the following command for single-gpu inference.

   .. code-block:: bash

      cd TinyLLaVA_Factory
      CUDA_VISIBLE_DEVICES=0 bash scripts/eval/textvqa.sh

POPE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1.	**Dataset:** Download COCO val2014_ and the coco_ folder that contains 3 json files, put them under ``path/to/your/dataset/eval/pope``.

.. _val2014: http://images.cocodataset.org/zips/val2014.zip
.. _coco: https://github.com/AoiDragon/POPE/tree/e3e39262c85a6a83f26cf5094022a782cb0df58d/output/coco

2. Please change ``MODEL_PATH``, ``MODEL_NAME``, ``EVAL_DIR``, and ``conv-mode`` in ``scripts/eval/pope.sh``.

3.	**Inference:** POPE does not support multi-gpus inference, please use the following command for single-gpu inference.

   .. code-block:: bash

      cd TinyLLaVA_Factory
      CUDA_VISIBLE_DEVICES=0 bash scripts/eval/pope.sh

MME
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1.	**Dataset:** Download the data following the official instructions here_.

.. _here: https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation

2. Please change ``MODEL_PATH``, ``MODEL_NAME``, ``EVAL_DIR``, and ``conv-mode`` in ``scripts/eval/mme.sh``.

3.	Downloaded images to ``MME_Benchmark_release_version``.

4.	put the official ``eval_tool`` and ``MME_Benchmark_release_version`` under ``path/to/your/dataset/eval/MME``.

5.	**Inference:** MME does not support multi-gpus inference, please use the following command for single-gpu inference.

   .. code-block:: bash

      cd TinyLLaVA_Factory
      CUDA_VISIBLE_DEVICES=0 bash scripts/eval/mme.sh

MM-Vet
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1.	**Datasets:** Extract mm-vet.zip_ to ``path/to/your/dataset/eval/mmvet``.

.. _mm-vet.zip: https://github.com/yuweihao/MM-Vet/releases/download/v1/mm-vet.zip

2. Please change ``MODEL_PATH``, ``MODEL_NAME``, ``EVAL_DIR``, and ``conv-mode`` in ``scripts/eval/mmvet.sh``.

3.	**Inference:** MM-Vet does not support multi-gpus inference, please use the following command for single-gpu inference.

   .. code-block:: bash

      cd TinyLLaVA_Factory
      CUDA_VISIBLE_DEVICES=0 bash scripts/eval/mmvet.sh
    
4.	Submit the results(``path/to/your/dataset/eval/mmvet/results``) to the mmvet_evaluation_server_.

.. _mmvet_evaluation_server: https://huggingface.co/spaces/whyu/MM-Vet_Evaluator

MMMU
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Datasets**: Extract MMMU.zip_ to ``path/to/your/dataset/eval/MMMU``.

.. _MMMU.zip: https://drive.google.com/file/d/1TJszQ23X-7TeMYDA7hVKpoHy9yo-lsc5/view?usp=sharing

2. Please change ``sample["img_path"]`` to your path in ``eval/download_images.py``，and download images as following.

   .. code-block:: bash

      cd path/to/your/dataset/eval/MMMU
      mkdir all_images
      python eval/download_images.py

3. Please change ``MODEL_PATH``, ``MODEL_NAME``, ``EVAL_DIR``, and ``conv-mode`` in ``scripts/eval/mmmu.sh``.

4. **Inference**: MMMU does not support multi-gpus inference, please use the following command for single-gpu inference.

   .. code-block:: bash

      cd TinyLLaVA_Factorysh
      CUDA_VISIBLE_DEVICES=0 bash scripts/eval/mmmu.sh
