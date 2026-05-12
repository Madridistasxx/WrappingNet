#!/bin/bash
python WrappingNet.py --gpus 0 --epochs 1000 --epochs_sphere 200 --latent_dim 512 --lr 1e-5 --data_name "manifold40" --model_name "LC"
uv run test.py --gpus 0 --epochs 10 --epochs_sphere 5 --latent_dim 512 --lr 1e-5 --data_name "tools" --model_name "LC"