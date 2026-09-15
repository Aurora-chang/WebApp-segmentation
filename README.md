# Pet segmentation webapp

A small Flask application for the Oxford-IIIT Pet binary segmentation assignment.
Choose a photo to view the original, green pet overlay, and downloadable black/white
PNG mask. It runs one CPU model in one Waitress process, with two HTTP threads and
two PyTorch compute threads. A lock admits only one photo-processing request at a
time; another receives a friendly busy response. No JavaScript or build step is needed.

## Verified deployment

Repository: https://github.com/Aurora-chang/WebApp-segmentation

The deployment owner verified the following on the actual VM:

| Component | Deployed configuration |
|---|---|
| Host | `aurora-vm`, Debian Linux |
| Python | **3.13.5** |
| Resources | 2 CPU cores, approximately 3.8 GB RAM; CPU-only inference |
| Virtual environment | `/home/aurora/WebApp-segmentation/.venv` |
| Trained checkpoint | `/home/aurora/gpu-jobs/j20260914150037g15y/best.pt` |
| Checkpoint input size | **256 × 256** |
| Server | Waitress, port 8000 |
| systemd service | `pet-segmentation-webapp.service` |
| HTTPS access | https://aurora-vm.monster-frog.ts.net (**tailnet-only**) |

The trained checkpoint loaded successfully on CPU. Real cat and dog uploads
produced segmentation results. Invalid/non-image uploads displayed readable errors
without crashing the application. The application remained running after SSH
disconnected. These are functional deployment checks, not a new accuracy benchmark.

## Install on aurora-vm

Python **3.13.5 is verified on Debian**; Python 3.12 was used for local macOS tests.
Use Python 3.13 with `venv` support to reproduce the VM setup. The commands below
are for a fresh installation; the working VM already has its virtual environment.

If the repository is not already present:

```bash
git clone https://github.com/Aurora-chang/WebApp-segmentation.git /home/aurora/WebApp-segmentation
```

Install into the virtual environment:

```bash
cd /home/aurora/WebApp-segmentation
python3.13 --version
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip check
python -c 'import torch, torchvision; print(torch.__version__, torchvision.__version__); assert torch.version.cuda is None, "Expected a CPU-only build"'
```

Installing the CPU wheels first keeps the manifest installation from fetching CUDA
packages on Linux. Do not start with only `pip install -r requirements.txt` in a
fresh Linux environment: the manifest alone does not select CPU builds.
The matching versions and CPU index follow the [PyTorch installation matrix](https://pytorch.org/get-started/previous-versions/).
For local macOS development, use `python -m pip install -r requirements.txt` directly
inside the virtual environment; omit the Linux CPU-index command.

## Provide the trained checkpoint

The deployed trained checkpoint is outside the checkout at
`/home/aurora/gpu-jobs/j20260914150037g15y/best.pt`. Keep it there and readable by
the service user `aurora`. A fresh deployment needs this artifact transferred
separately; cloning the repository does not supply weights. Do not add weights
to GitHub. `.gitignore` excludes `*.pt`, `*.pth`, `*.ckpt`, and `checkpoints/`.

```bash
export PET_CHECKPOINT=/home/aurora/gpu-jobs/j20260914150037g15y/best.pt
```

The application loads the checkpoint **once at startup**, using
`torch.load(..., map_location="cpu", weights_only=True)`, then creates
`PetUNet(pretrained=False)`, strictly loads `checkpoint["model"]`, and calls `eval()`.
No pretrained download occurs. Use your own training artifact; checkpoint upload
is not part of the web interface.

The checkpoint must contain the training dictionary, including `model` and
`config.size`. Missing, corrupt, or incompatible checkpoints stop startup with a
clear error. This small-VM implementation accepts integer sizes from 64 to 512;
it fails explicitly for other sizes instead of substituting a resolution. Training
defaults to 256, and the deployed checkpoint confirms size 256, but **the actual
checkpoint value is always used**.

`PET_CHECKPOINT` may point to another compatible checkpoint. Alternatively, pass
`--checkpoint /absolute/path/to/best.pt`, which overrides the environment variable.
The dependency manifest describes the webapp environment, not the original training
environment; exact installed VM package versions can be inspected with
`.venv/bin/python -m pip freeze`.

## Run with systemd

The deployed application is managed by `pet-segmentation-webapp.service`. To
reproduce it, create `/etc/systemd/system/pet-segmentation-webapp.service` with
`sudoedit` and the following minimal unit. This is a reproduction template using
the verified paths, not a captured copy of the existing VM unit:

```ini
[Unit]
Description=Pet segmentation web application
After=network.target

[Service]
Type=simple
User=aurora
WorkingDirectory=/home/aurora/WebApp-segmentation
Environment=PET_CHECKPOINT=/home/aurora/gpu-jobs/j20260914150037g15y/best.pt
ExecStart=/home/aurora/WebApp-segmentation/.venv/bin/python /home/aurora/WebApp-segmentation/app.py --host 127.0.0.1 --port 8000
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable and start the service, then inspect status and logs:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now pet-segmentation-webapp.service
sudo systemctl status pet-segmentation-webapp.service --no-pager
sudo journalctl -u pet-segmentation-webapp.service -n 50 --no-pager
curl -I http://127.0.0.1:8000
```

After changing an existing unit, run `daemon-reload` followed by
`sudo systemctl restart pet-segmentation-webapp.service`. Inspect the installed
unit with `sudo systemctl cat pet-segmentation-webapp.service` before editing it.
systemd manages the process independently of SSH; enabling the unit also starts
it at boot. See the [systemd service reference](https://www.freedesktop.org/software/systemd/man/systemd.service.html).

For foreground troubleshooting when the service is stopped, run:

```bash
cd /home/aurora/WebApp-segmentation
PET_CHECKPOINT=/home/aurora/gpu-jobs/j20260914150037g15y/best.pt .venv/bin/python app.py --host 127.0.0.1 --port 8000
```

Do not run this alongside the service on the same port. Waitress runs without
Flask's debug mode or reloader and loads one model per process.

## Tailscale Serve and remote access

Verified final URL: **https://aurora-vm.monster-frog.ts.net**

```text
Browser on the tailnet
  -> https://aurora-vm.monster-frog.ts.net
  -> Tailscale Serve
  -> http://127.0.0.1:8000 (Waitress / Flask)
  -> shared CPU inference
```

On a VM already connected to the intended tailnet, with the application running,
configure background HTTPS proxying:

```bash
sudo tailscale serve --bg http://127.0.0.1:8000
sudo tailscale serve status
```

Follow any HTTPS-enablement prompt if this is the first Serve setup on the tailnet.
The `--bg` setting persists beyond the command's shell session. See the
[Tailscale Serve reference](https://tailscale.com/docs/reference/tailscale-cli/serve).

The URL is **tailnet-only**, not a public internet endpoint. Open it from a device
connected to the tailnet and permitted by its access policy. Tailscale supplies
HTTPS; Waitress serves local HTTP. This setup uses Serve, not Funnel, and does not
require exposing VM port 8000 publicly. A different tailnet or machine name will
produce a different HTTPS hostname.

## Exact training/inference contract

`model.py` is copied byte-for-byte from the previous `pet-segmentation` project:
a custom U-Net with ResNet-18 encoder and one output logit channel. All uploaded
photos go through `Segmenter.predict()` in **inference.py**, the only runtime
preprocessing implementation:

1. Pillow decodes and converts to `RGB`; no BGR swap, EXIF transpose, or alpha compositing.
2. Read `size` from `checkpoint["config"]["size"]`.
3. Resize the **PIL image** directly to `[size, size]` with
   `TF.resize(..., InterpolationMode.BILINEAR)`, exactly as before. No crop or padding;
   no explicit antialias override. Nonsquare images are stretched for model input.
4. `TF.to_tensor()` converts RGB bytes to CHW float values scaled to `[0, 1]`.
5. Normalize in RGB order with mean `[0.485, 0.456, 0.406]` and std
   `[0.229, 0.224, 0.225]`; add a batch dimension. Input is `[1, 3, size, size]`.
6. Execute on CPU inside `torch.inference_mode()`.
7. Bilinearly resize **logits** back to `(original height, original width)` using
   `align_corners=False`, then apply sigmoid and threshold `>= 0.5`.
8. Save/display the single-channel mask as 255 for pet and 0 for background.
   The overlay uses the previous script's green tint blended at 0.4 within the mask.

Training-only horizontal flips and color jitter are omitted, as in the original
`predict.py`. Both cat and dog are foreground. Trimap label 3 was ignored during
training/evaluation; there is no ignored-pixel mask for uploaded photos.

Original and overlay previews are JPEG encoded for transfer size; the downloadable
mask is lossless PNG. These display encodings happen **after** inference. Results
retain original decoded dimensions and orientation.

## Upload handling and resource limits

- The whole multipart request is limited to **10 MiB**, including form overhead.
  Both Flask and Waitress enforce this. Flask returns a styled 413 error; requests
  rejected earlier by Waitress receive its plain 413 response.
- Actual content must decode as a still JPEG, PNG, or WebP image with at most
  **8 million pixels**. File extensions and browser MIME types are not trusted.
- Missing, empty, unsupported, animated, corrupt, and oversized images produce
  readable errors. Unexpected inference errors are logged server-side and return
  a generic error; the processing lock is released so later uploads can proceed.
- The app stores no uploaded photos or results permanently. Multipart handling may
  spool request data to temporary files; responses embed re-encoded images directly
  and use `Cache-Control: no-store`.
- One model/process and one active prediction bound RAM usage. Do not add multiple
  worker processes on the 2-core, 3.8-GB VM. Functional CPU deployment is verified;
  no quantitative latency or peak-RAM benchmark is recorded here.
- Checkpoints, virtual environments, Python/test caches, local logs, and runtime
  upload/result directories are excluded by `.gitignore`. Service logs normally
  go to the system journal, outside the repository.

## Tests and sanity checks

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m pip check
git diff --check
```

Tests use a temporary synthetic checkpoint with real PetUNet weights, never a
download or the training dataset. They check the original preprocessing equations
and resize behavior, RGB/grayscale/RGBA handling, CPU/eval loading, strict checkpoint
validation, inference mode, logit resize/threshold order, real forward passes,
successful web uploads, single model reuse, upload failures, and busy/error recovery.
Synthetic weights establish execution and compatibility assumptions, not accuracy.

Local verification completed with Python 3.12, torch 2.8.0, torchvision 0.23.0,
Pillow 12.3.0, Flask 3.1.3, and Waitress 3.0.2 on macOS:

- All 16 unittest cases passed.
- A real Waitress process served the page, stylesheet, and a successful multipart
  upload using a synthetic checkpoint with `config.size=256`; it was stopped afterward.
- `pip check` passed, and `cmp` confirmed the model file is unchanged.

To verify the copied model when the previous repository is available:

```bash
cmp model.py ../pet-segmentation/model.py
```

## Files

- `model.py`: unchanged architecture from the training project.
- `inference.py`: shared decoding, preprocessing, checkpoint loading, prediction, overlay.
- `app.py`: Flask routes, bounded processing, startup CLI, Waitress server.
- `templates/index.html`, `static/style.css`: responsive upload and results page.
- `requirements.txt`: runtime dependencies; tests use standard-library unittest.
- `tests/test_app.py`: inference and HTTP regression tests.
- `.gitignore`: excludes weights, virtual environments, and generated files.

Upload limits follow [Flask's upload documentation](https://flask.palletsprojects.com/en/stable/patterns/fileuploads/);
HTTP thread and request limits use [Waitress settings](https://docs.pylonsproject.org/projects/waitress/en/stable/arguments.html).
