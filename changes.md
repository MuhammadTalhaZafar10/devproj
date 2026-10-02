### Training Parameter Changes

The following training parameters were adjusted to improve the model's learning performance and generalization:

* **Learning Rate:** Changed from `0.001` to `0.0005` to allow more gradual model updates.
* **Batch Size:** Increased from `32` to `64` to provide more stable gradient updates.
* **Number of Epochs:** Increased from `20` to `50` to give the model additional training iterations.
* **Optimizer:** Changed to `Adam` for adaptive learning-rate optimization.
* **Dropout:** Set to `0.3` to reduce overfitting and improve generalization.
* **Validation Split:** Set to `20%` of the training data for validation during training.

These parameters can be modified according to the dataset size, model architecture, and training performance. Model accuracy, validation loss, and training loss should be monitored after each experiment to determine the effect of the parameter changes.
