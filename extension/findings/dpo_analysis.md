# Why DPO did not help

DPO trains on (chosen, rejected) pairs built from the untrained model's own samples. Mean of each signal over a few windows of training steps (from the DPO training log), next to the validation accuracy of the closest checkpoint:

| Signal | steps 1 to 10 | steps 91 to 110 | steps 291 to 310 | steps 486 to 495 |
|---|---|---|---|---|
| Preference accuracy (chosen scored above rejected) | 0.506 | 0.838 | 0.906 | 0.885 |
| Reward margin (chosen minus rejected) | 0.088 | 1.489 | 2.790 | 3.489 |
| Log-probability of the chosen answer | -27.242 | -34.108 | -38.762 | -44.577 |
| Log-probability of the rejected answer | -27.258 | -48.405 | -65.411 | -79.535 |
| Validation EX of the checkpoint (%) | 61.9 (step 0) | 57.9 (step 100) | 44.3 (step 300) | 41.1 (step 495) |

What the numbers show:
- The model gets better at the DPO objective: it scores the chosen answer above the rejected one more and more often, and the margin between them grows.
- But the log-probability of the chosen (correct) answers also goes down. The rejected answers go down faster, which is enough to lower the loss, but the model becomes less likely to write the correct SQL it was shown. This is called likelihood displacement.
- Validation accuracy falls from the first checkpoint on, so the best DPO checkpoint is step 100, and it is still slightly below the untrained model on every test set.
- GRPO and RLOO do not have this problem because they keep sampling new answers from the current model and reward the ones that actually run correctly, while DPO only sees fixed pairs from the untrained model.