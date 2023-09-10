# OpenAUC: Towards AUC-Oriented Open-Set Recognition
This is a Pytorch implementation of our paper: [OpenAUC: Towards AUC-Oriented Open-Set Recognition](https://arxiv.org/abs/2210.13458). **If you only want to evaluate the model performance on OpenAUC,** please refer the file `utils/test_utils`. And a more detailed instruction can be found in our [library](https://github.com/statusrank/XCurve/blob/master/example/example_ipynb/Metrics_for_AUTKC_OpenAUC.ipynb).

> **Abstract:** Traditional machine learning follows a close-set assumption that the training and test set share the same label space. While in many practical scenarios, it is inevitable that some test samples belong to unknown classes (open-set). To fix this issue, Open-Set Recognition (OSR), whose goal is to make correct predictions on both close-set samples and open-set samples, has attracted rising attention. In this direction, the vast majority of literature focuses on the pattern of open-set samples. However, how to evaluate model performance in this challenging task is still unsolved. In this paper, a systematic analysis reveals that most existing metrics are essentially inconsistent with the aforementioned goal of OSR: (1) For metrics extended from close-set classification, such as Open-set F-score, Youden's index, and Normalized Accuracy, a poor open-set prediction can escape from a low performance score with a superior close-set prediction. (2) Novelty detection AUC, which measures the ranking performance between close-set and open-set samples, ignores the close-set performance. To fix these issues, we propose a novel metric named OpenAUC. Compared with existing metrics, OpenAUC enjoys a concise pairwise formulation that evaluates open-set performance and close-set performance in a coupling manner. Further analysis shows that OpenAUC is free from the aforementioned inconsistency properties. Finally, an end-to-end learning method is proposed to minimize the OpenAUC risk, and the experimental results on popular benchmark datasets speak to its effectiveness.

Our codes are based on the repositories [Open-Set Recognition: a Good Closed-Set Classifier is All You Need?](https://github.com/sgvaze/osr_closed_set_all_you_need) and [Learning Placeholders for Open-Set Recognition](https://github.com/zhoudw-zdw/CVPR21-Proser).

## Dependencies
Please refer to the `requirements.yml` in the root folder.

## Settings
The parameters are stored in the file `utils/config.py`.

The datasets can be found in the README of [Open-Set Recognition: a Good Closed-Set Classifier is All You Need?](https://github.com/sgvaze/osr_closed_set_all_you_need). After downloading the datasets, please put them in the folder `data`.

Please put pre-trained models in the folder `models`. And all the outputs will be stored in the folder `log` with a unique id.

We provide the bash script in the folder `bash_scripts` for the experiments on the *CUB* dataset. Note that here we follow the traditional assumption that $\forall c \in Y_k, \mathbb{P}[y = c \mid x]$ and $r(x) \propto 1 / \max_{k \in Y_k} f(x)_k$, which shows similar performances as those reported in our paper.


## Citation

```
@InProceedings{openauc,
    title = {OpenAUC: Towards AUC-Oriented Open-Set Recognition},
    author = {Zitai Wang and  Qianqian Xu and Zhiyong Yang and Yuan He and Xiaochun Cao and Qingming Huang},
    booktitle = {Annual Conference on Neural Information Processing Systems},
    year = {2022},
    pages = {25033--25045}
}
```