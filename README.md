# Code for Learning Multilingual Topics with Neural Variational Inference

[NLPCC2020 paper](https://link.springer.com/chapter/10.1007/978-3-030-60450-9_66)

## Usage

### 0. Download datasets

[Google Drive](https://drive.google.com/drive/folders/15Ywhv_O1h8bysjqbG8ZtS7cnGyz06yUr?usp=sharing)

Unzip and move the datasets to ./data

### 1. Prepare environment

    python==3.6
    tensorflow-gpu==1.13.1
    scipy==1.5.2
    scikit-learn==0.23.2 

### 2. Training

    python run.py --data_dir data/{dataset} --output_dir output/{dataset}

### 3. Evaluation

1. Topic coherence: [topic_interpretability](https://github.com/jhlau/topic_interpretability)

2. Topic diversity:

        python utils/TU.py --data_path {path of topic word file}

3. Classification: use the SVM of scikit-learn.

## Citation
If you want to use our code, please cite as

    @inproceedings{Wu2020,
        author = {Wu, Xiaobao and Li, Chunping and Zhu, Yan and Miao, Yishu},
        booktitle = {International Conference on Natural Language Processing and Chinese Computing},
        title = {{Learning Multilingual Topics with Neural Variational Inference}},
        year = {2020}
    }
