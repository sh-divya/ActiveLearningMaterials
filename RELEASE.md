# How to make a Dave Release for GflowNets

1. Decide on a release name
   1. Typically: `{version}-{model}-{data}`

2. Make a release directory in `/network/scratch/s/schmidtv/Public/crystals/proxy-ckpts`
   ```bash
   mkdir /network/scratch/s/schmidtv/Public/crystals/proxy-ckpts/{release_name}
   ```
3. Copy **a single checkpoint** in there
   ```bash
   # aussumig you are in the release dir
   cp path/to/epoch=91-step=23460-total_val_mae=0.1337.ckpt .
   ```
4. Add the wandb url as a text file in the release directory
   ```
   echo "https://wandb.ai/mila-ocp/Dave-MBform/runs/ywrwnf1b/overview" > wandb_url.txt
   ```
5. Make sure your model will load properly with `gflownet`:
    ```python
    from dave import prepare_for_gfn

    model, proxy_loaders, scales = prepare_for_gfn(
        {"mila": "/network/scratch/s/schmidtv/Public/crystals/proxy-ckpts"},
        "{release_name}",
        True,
    )
    ```
7. If all goes well at this point, **create a release on this repo**
8. Create a new branch on the  `gflownet` repo and update `config/proxy/dave.yaml:release` to point to your release
   1. Make sure your proxy works with
   ```
   (gflownet-env) $ python main.py user=$USER +experiments=neurips23/crystal-comp-sg-lp.yaml device=cpu logger.test.top_k=-1 gflownet.optimizer.batch_size=100 logger.do.online=False
   ```
9. Create a PR on the `gflownet` repo with your new branch
   2. **If your model requires new packages**, it should be made very clear when creating this PR to `gflownet`
   3. For instance in the PR comments `## 💥 Breaking changes: new packages`
