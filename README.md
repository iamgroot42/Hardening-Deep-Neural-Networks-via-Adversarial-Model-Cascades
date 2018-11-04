# Hardening Deep Neural Networks _via_ Adversarial Model Cascades 

### Working

- Adversary gets hold of unlabelled data (not overlapping with target)
- Make queries to black-box model, use results to create dataset.
- Using this collected data, train a proxy network.
- Generate adversarial examples for the trained proxy network.
- Calculate error rate on the target network, for images generated by the proxy adversarial network.

### Attacks present

- FGSM
- VAP
- PGM
- EAP

### Setting it up

- `bash prepare.sh` to download required data and models
- `python Code/test_accuracy.py --model_path <target_model> --dataset <dataset>` to get test accuracy ,from Code/ folder 
- `bash test*.sh <dataset> <target_model> <proxy_model>`, where * denotes anoy of the 7 attacks given in the repo ,from the Scripts/ folder
- For the basic bagging setup, run `bash genericBagging.sh <dataset> <path_to_seed_model> <new_folder_for_bag>    <path_to_file_containing_order_of_attacks> <transfer_parameters_per_bag?>` ,from the Scripts/ folder

For example, `bash genericBagging.sh mnist PlainModel MYBAG/ ORDER no`
- For the adaptive bagging setup, run `bash adaptiveBagging.sh <dataset> <path_to_seed_model> <new_folder_for_bag> <path_to_file_containing_order_of_attacks> <transfer_parameters_per_bag?> <path_to_proxy_model>`

For example, `bash adaptiveBagging.sh mnist PlainModel MYBAG/ ORDER no ProxyNormal`
- For testing bagging on your own attack data, run `python ../Code/bagging.py --mode test --dataset <dataset> --model_dir <model_bag_directory> --data_x <data_X> --data_y <data_Y>  --predict_mode <voting/weighted>`

