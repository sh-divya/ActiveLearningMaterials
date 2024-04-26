# How to make a Dave Release for GflowNets

1. Decide on a release version
   1. Typically: `0.{model_version_number}.{minor_update_number}`

2. Make a release directory in `/network/scratch/s/schmidtv/crystals-proxys/proxy-ckpts/`

   ```bash
   mkdir /network/scratch/s/schmidtv/crystals-proxys/proxy-ckpts/{release_name}
   ```

4. Copy **a single checkpoint** in there

   ```bash
   # aussumig you are in the release dir
   cp path/to/epoch=91-step=23460-total_val_mae=0.1337.ckpt .
   ```

5. Add the wandb url as a text file in the release directory

   ```bash
   echo "https://wandb.ai/mila-ocp/Dave-MBform/runs/ywrwnf1b/overview" > wandb_url.txt
   ```

6. Make sure your model will load properly with `gflownet`:

    ```python
    from dave import prepare_for_gfn

    model, proxy_loaders, scales = prepare_for_gfn(
        {"mila": "/network/scratch/s/schmidtv/crystals-proxys/proxy-ckpts/"},
        "{release_name}",
        True,
    )
    ```

7. **IF you added a dependency: add it to `pyproject.tom`**
   1. DAVE will be installed with `pip install .`

8. If all goes well at this point, **create a release on this repo**
9.  Create a new branch on the  `gflownet` repo and update `config/proxy/dave.yaml:release` to point to your release
   1. Make sure your proxy works with

   ```bash
   (gflownet-env) $ python main.py user=$USER env=crystals/crystal proxy=crystals/dave gflownet=trajectorybalance device=cpu logger.do.online=False logger.project_name=playground logger.test.period=100 gflownet.optimizer.n_train_steps=10000 gflownet.random_action_prob=0.1 logger.test.n=100
   ```

10. Create a PR on the `gflownet` repo with your new branch
   1. **If your model introduces breaking changes**, it should be made very clear when creating this PR to `gflownet`
   2. For instance in the PR comments `## 💥 Breaking changes: {something}`
