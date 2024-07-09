# Source-inference-FL
Forked Repository from SIA project of FL to custom algorithm on Federated Learning.

# Implementation
You can run the following code to implement the source inference attacks. The datasets provided in this rep are `Synthetic` and `MNIST` datasets. For the `MNIST` dataset, it will automatically be downloaded. For `Synthetic` dataset, please first run the following code to generate it:
```python
python generate_synthetic.py
```

You can try different `--alpha` (data distribution), `--num_users`(number of parties), `--local_ep` (number of local epochs) to see how the attack performance changes. For `Synthetic` dataset, we set `--model=mlp`. For `MNIST` dataset, we set `--model=cnn`.
```python
python main_fed.py --dataset=MNIST --model=cnn --alpha=1 --num_users=6 --local_ep=5
```
