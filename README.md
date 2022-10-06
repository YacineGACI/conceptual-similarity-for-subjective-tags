# Conceptual Similarity for Subjective Tags
This implementation contains Pytorch code for conceptual similarity that works on subjective tags. It also contains code related to the automatic generation of sythetic datasets to train conceptual similarity models.

You can find the lists of seeds in **'data/dataset_generation/seeds.txt'** and the parameter configuration of the different seed expansion methods in **'data/dataset_generation/expansion.config.json'**

You can play around the seeds and the parameter configurations by either modifying those files or create your own. However, you mist ensure that the files are formatted as in **'data/dataset_generation/seeds.txt'** for the seeds and **'data/dataset_generation/expansion.config.json'** for the configuration.


## Dataset Generation
To generate a new dataset in order to train a conceptual similarity model for subjective tags, there is a need to specify the seeds, the parameter configuration of seed expansion methods, and a filename to the output. Then, run the following:
`python -m dataset.create_dataset --seeds <path_to_seeds> --expansion_config <path_to_expansion_config> --output_filename <path_to_dataset_output>`
There are other parameters that you can use in order to adjust the dataset generation process, e.g.:
- *dataset_size*: To specify the number of lines in the fial dataset
- *min_ratio_pos*: The minium ratio of positive examples (examples with a label of 1)
- *as_min_consensus_rate*: The percentage of expandors to suggest a given expansion for the expansion to be considered correct, and included in the final set.



## Training the conceptual similarity model
To train the conceptual similarity model, run the following:
`python train.py --data <path_to_dataset_> --output <path_to_model_output>`
There are also other parameters to configure training, such as:
- *lr*: for the learning rate
- *epochs*: for the number of epochs
- *batch_size*
- *dropout*
- *weight_decay* ...



## Evaluation
### Evaluating the conceptual similarity model
After training is complete, you can evaluate the performance of the similarity model by:
`python -m evaluation.evaluate_similarity --data <path_to_folder_containing_eval_data> --model <path_to_model>`



### Adding noise to the dataset 
In order to introduce noise to the dataset, run the following:
`python evaluation/add_noise.py --dataset <path_to_dataset> --output <path_to_new_dataset> --noise_ratio <ratio>`
Then, train another similarity model with the noisy dataset, evaluate it as above, and observe the effect that noise introduces.