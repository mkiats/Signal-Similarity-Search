## Quick Start

1. Open a terminal and run `pip install -r requirements.txt`
2. Open `data_separation.ipynb` and run all blocks to separate `data.zip` into labelled and unlabelled data folders.
3. You may wish to visualise the data distribution in `data_distribution.ipynb`.
4. Run `python initial_training.py` to start training the model. The hyperparameters can be tuned within `initial_training.py`. Once the training is completed, the model's weights should be saved in the `results` folder.
5. Run `python finetune.py` to start finetuning the model. The pretrained model to finetune can be changed within `finetune.py`, along with the hyperparameters for finetuning.
