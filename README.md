# Certified Defenses for Data Poisoning Attacks

**NOTE: I highly recommend that you do not try and use this as a python package in its current state!**

This fork is for me to make misc changes tailored to my own workflow. See the original paper for full details.

> Jacob Steinhardt, Pang Wei Koh, and Percy Liang
>
> [Certified Defenses for Data Poisoning Attacks](https://arxiv.org/abs/1706.03691)
>
> _NIPS_ 2017.

## What does everything do?
To keep track of and make sense of what things do. I have also moved anything that did not seem strictly necessary to the legacy folder.

### Data / Dataset
- **data_utils.py**: Misc data utilities. E.g. Determine centroids. Get sphere and slab thresholds from percentile. Project data onto feasible set. Filter data outside of feasible set.
- **datasets.py**: Dataset loading and saving utilities. E.g. Loading datasets. Creating paths to save data from intermediate steps. Setting *some* free parameters.

### Defenses
- **defenses.py**: Sphere and slab defenses?

### Benchmarks
- **generate_label_flip_baseline.py**: Generates poisoned dataset using the label flip attack.

### Visualisation
- **plotter.py**: Utility to setup the plot as in Figure 1.

### Bounds
- **generate_or_process_bounds.py**: Depending on the particular setting, this *either* calls the functions to generate the upper and lower bounds (lower bound being any realised attack). Or simply attempts to load the results, assuming that they already have been run.
    - **Oracle**:
    - **Data Dependent**:
    - **Label Flip**
    - **Integer Constrained**:
- **upper_bounds.py**:

## Common Data Formats

### Bound
The **bound** format is used to store the results of the upper and lower bound calculation. Created in **generate_or_process_bounds.py**.

Variable | Type | Description
--- | --- | ---
percentile | np.ndarray of shape () | Percentile of data to keep when setting the outlier removal threshold free parameter.
weight_decay | np.ndarray of shape () | The SVM is trained such that the [upper_bound_norm_square] is equal to some value. The weight decay is used to calculate C by  1/(weight_decay * num_instances) and therefore C can be recovered from this data.
epsilons | np.ndarray of shape (epsilons,) | Fraction of poisoning data added.
upper_total_losses | np.ndarray of shape (epsilons,) | Upper bound loss on the poisoned dataset due to both clean and malicious data.
upper_good_losses | np.ndarray of shape (epsilons,) | Upper bound loss on only clean data.
upper_bad_losses | np.ndarray of shape (epsilons,) | Upper bound loss on only malicious data.
upper_good_acc | np.ndarray of shape (epsilons,) | Upper bound accuracy on only clean data.
upper_bad_acc | np.ndarray of shape (epsilons,) | Upper bound accuracy on only malicious data (100% if [worst_margin] > 0, 0% otherwise)
upper_params_norm_sq | np.ndarray of shape (epsilons,) | Used to determine SVM C. ???
lower_total_train_losses | np.ndarray of shape (epsilons,) | Lower bound training loss on the poisoned dataset due to both clean and malicious data.
lower_avg_good_train_losses | np.ndarray of shape (epsilons,) | Lower bound training loss on only clean data.
lower_avg_bad_train_losses | np.ndarray of shape (epsilons,) | Lower bound training loss on only malicious data.
lower_test_losses | np.ndarray of shape (epsilons,) | Lower bound testing loss.
lower_overall_train_acc | np.ndarray of shape (epsilons,) | Lower bound training accuracy on poisoned dataset due to both clean and malicious data.
lower_good_train_acc | np.ndarray of shape (epsilons,) | Lower bound training accuracy on only clean data.
lower_bad_train_acc | np.ndarray of shape (epsilons,) | Lower bound training accuracy on only malicious data.
lower_test_acc | np.ndarray of shape (epsilons,) | Lower bound testing accuracy.
lower_params_norm_sq | np.ndarray of shape (epsilons,) | Same as *upper_params_norm_sq*, but for lower bound.
lower_weight_decays | np.ndarray of shape (epsilons,) | Same as *weight_decay*, but for lower bound.

### Attack
Created in **generate_or_process_bounds.py**.

Variable | Type | Description
--- | --- | ---
X_modified | np.ndarray of shape (instances, dimensions) | Poisoned training features.
Y_modified | np.ndarray of shape (instances,) | Poisoned training lables.
X_test | np.ndarray of shape (instances, dimensions) | Testing features.
Y_test | np.ndarray of shape (instances,) | Testing labels.
idx_train | | ???
idx_poison | | Indices of malicious data in the poisoned dataset.

### Poisoned Dataset
Created in **generate_label_flip_baseline.py**.

Variable | Type | Description
--- | --- | ---
poisoned_X_train | np.ndarray of shape (instances, dimensions) | Poisoned training features.
Y_train | np.ndarray of shape (instances,)| Poisoned training labels.

### Dataset
Created in **generate_label_flip_baseline.py**.

Variable | Type | Description
--- | --- | ---
X_train | np.ndarray of shape (instances, dimensions) | Training features.
Y_train | np.ndarray of shape (instances,) | Training labels.
X_test | np.ndarray of shape (instances, dimensions) | Testing features.
Y_test | np.ndarray of shape (instances,) | Testing labels.