# Custom Federated Learning - Series Based Learning
Forked Repository from SIA project of FL to custom algorithm on Federated Learning.

# Implementation

You can try different `--alpha` (data distribution), `--num_users`(number of parties), `--local_ep` (number of local epochs) to see how the attack performance changes. For `MNIST` dataset, we set `--model=cnn`.
```python
python main_fed.py --dataset=MNIST --model=cnn --alpha=1 --num_users=6 --local_ep=5
```
