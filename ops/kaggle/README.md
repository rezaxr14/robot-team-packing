# Publishing the base model from Kaggle (never from the laptop)

The 9.4 GB Qwen3.5-4B snapshot must not be uploaded from this machine. `publish_model.py` runs as a
private Kaggle kernel: Kaggle downloads the model to its own disk and publishes it as the private
dataset `rezanadimikj/rtp-model-qwen35-4b-851bf6e8`. That slug is exactly what
`rtp.remotes.RemoteStore.resolve()` looks for, so the next notebook attaches it at `/kaggle/input`
and never touches Kaggle's limited working directory.

```
ops/kaggle_model.py push      # upload the ~4 KB kernel and start the run
ops/kaggle_model.py status    # KernelWorkerStatus.QUEUED / RUNNING / COMPLETE / ERROR
ops/kaggle_model.py output    # the kernel log
ops/kaggle_model.py verify    # isPrivate + id of the resulting dataset
```

All four go through `http://localhost:10808` (`RTP_KAGGLE_PROXY`) with retries; Kaggle refuses the
direct connection on this network.

## One manual step (only the account owner can do it)

A Kaggle kernel has **no Kaggle API credentials of its own** — the first attempt failed exactly
there:

```
File "/kaggle/src/script.py", line 41, in creds
    token = UserSecretsClient().get_secret("KAGGLE_API_TOKEN")
ConnectionError: Connection error trying to communicate with service.
```

To let the kernel create the dataset, add a secret to the notebook:

1. open <https://www.kaggle.com/code/rezanadimikj/rtp-publish-qwen35-4b/edit>
2. **Add-ons → Secrets → Add a secret** named `KAGGLE_API_TOKEN`
3. paste the same token that is in `.env` (`KAGGLE_API_TOKEN`), save
4. `ops/kaggle_model.py push` again (a new version of the kernel re-runs the script)

`HF_TOKEN` is *not* needed: the base model is public. If the download ever needs auth, add it as a
second secret with the same name.

## What the kernel does

1. reads credentials (env, else the secret);
2. `snapshot_download(Qwen/Qwen3.5-4B, revision=851bf6e8)` straight into the dataset folder;
3. materialises any hub symlinks and drops the hub bookkeeping dir (so the upload is plain files);
4. writes `dataset-metadata.json` with `isPrivate: true`;
5. `dataset_create_new` (or `dataset_create_version` if it already exists).

The kernel is `is_private: true`; nothing about this run is public.
