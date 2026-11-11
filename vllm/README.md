# MiniInfer vLLM setup

This project runs vLLM through Ubuntu on WSL2. vLLM does not support native
Windows installations. The setup uses a separate Python 3.12 environment in
your Linux home directory and serves a small default model suitable for a
4 GB GPU.

## 1. Finish WSL2 installation

The Windows features required by WSL2 have been enabled. Restart Windows to
activate them, then open **PowerShell as Administrator** and run:

```powershell
wsl --install -d Ubuntu
```

Restart again if prompted. Open Ubuntu from the Start menu and finish its
first-run setup by creating a Linux username and password.

The Windows NVIDIA driver provides CUDA support inside WSL. Do not install a
Linux NVIDIA display driver inside Ubuntu.

## 2. Install vLLM

In the Ubuntu terminal, run:

```bash
cd "/mnt/c/Users/frank/Desktop/3. Projects/10. MiniInfer"
bash setup-vllm.sh
```

Before installing vLLM, install the compiler used by its CUDA kernels:

```bash
sudo apt-get update
sudo apt-get install -y build-essential
```

The setup checks that WSL can see the GPU and compiler, creates an isolated
Python 3.12 environment under `~/.venvs/miniinfer-vllm`, installs vLLM, and
verifies CUDA. The first install downloads several gigabytes of packages.

## 3. Start the model server

In Ubuntu:

```bash
cd "/mnt/c/Users/frank/Desktop/3. Projects/10. MiniInfer"
bash serve-vllm.sh
```

The default model is `Qwen/Qwen2.5-0.5B-Instruct`; its weights are downloaded
on first start and cached in your Linux home directory. The server listens on
`127.0.0.1:8000` and is reachable from Windows at `http://localhost:8000`.
It has no authentication, so keep it bound to localhost.

Test it from a second Ubuntu terminal:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-0.5B-Instruct","messages":[{"role":"user","content":"Hello!"}],"max_tokens":64}'
```

The server limits context length and concurrent sequences to reduce VRAM use.
If you change the model, set `MODEL` before running `serve-vllm.sh`; larger
models may not fit in 4 GB of VRAM.

For official requirements and installation details, see the
[vLLM GPU installation guide](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/).
