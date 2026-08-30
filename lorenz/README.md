# lorenz

A side project, independent of the `lift` package next to it: simulate the
Lorenz attractor with diffrax, then train a small causal transformer to
predict the trajectory forward.

```bash
pip install -e '..[lorenz]'

python lorenz.py                  # simulate 64 trajectories -> lorenz_dataset.npz
python transformer.py             # train, checkpointing into checkpoints/
python transformer.py --resume    # continue from the latest checkpoint
python transformer.py --eval      # plot a prediction against ground truth
```

Both scripts read and write relative to this directory. The dataset and
checkpoints are gitignored.
