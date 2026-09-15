# Pet segmentation webapp

A small Flask application for the Oxford-IIIT Pet binary segmentation assignment.
Choose a photo to view the original, green pet overlay, and downloadable black/white
PNG mask. It runs one CPU model in one Waitress process, with two HTTP threads and
two PyTorch compute threads. A lock admits only one photo-processing request at a
time; another receives a friendly busy response. No JavaScript or build step is needed.

## Install on aurora-vm

Use Python 3.10–3.12 (3.12 was used for local checks), with `venv` available.
From the repository directory on the VM:

```bash
cd ~/WebApp-segmentation
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip check
```

Adjust the `cd` path if your checkout lives elsewhere. Installing the CPU wheels
first keeps the manifest installation from fetching CUDA packages on Linux.
The matching versions and CPU index follow the [PyTorch installation matrix](https://pytorch.org/get-started/previous-versions/).
For local macOS development, use `python -m pip install -r requirements.txt` directly
inside the virtual environment; omit the Linux CPU-index command.

## Provide the trained checkpoint

Keep your trained `best.pt` outside this checkout, for example at
`/home/aurora/models/pet-segmentation/best.pt`. Copy the artifact from your completed
training job to that location yourself when preparing the VM. Do not add weights
to GitHub. `.gitignore` excludes `*.pt`, `*.pth`, `*.ckpt`, and `checkpoints/` as well.

```bash
export PET_CHECKPOINT=/home/aurora/models/pet-segmentation/best.pt
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
defaults to 256, but **the actual checkpoint value is always used**.

The trained checkpoint and its `environment.json` are not present locally. The
dependency versions here are the webapp environment, not a claim about the original
training environment. Real checkpoint compatibility and segmentation quality must
be checked when that artifact is available.

## Start the server

Recommended command on aurora-vm, after installation and placing the checkpoint:

```bash
PET_CHECKPOINT=/home/aurora/models/pet-segmentation/best.pt .venv/bin/python app.py --host 0.0.0.0 --port 8000
```

Alternatively, supply `--checkpoint /absolute/path/to/best.pt`; it overrides the
environment variable. The default host is `127.0.0.1` for local-only use. Waitress
runs without Flask's debug mode or reloader, so startup does not load two models.
Stop with Ctrl+C. No system service or Tailscale configuration is included.

### Access from another machine

With the server bound to `0.0.0.0`, open `http://<VM-IP>:8000` in your browser.
Use the VM's reachable address (not `0.0.0.0`). This requires an existing network
route and permission for inbound TCP port 8000; this project changes neither.
The app has no authentication or TLS, so use it on your private assignment network.

If SSH is your available route, start the server on its default loopback address:

```bash
PET_CHECKPOINT=/home/aurora/models/pet-segmentation/best.pt .venv/bin/python app.py
```

Then run this on the other machine and open `http://127.0.0.1:8000` there:

```bash
ssh -N -L 8000:127.0.0.1:8000 aurora@<VM-IP>
```

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
  worker processes on the 2-core, 3.8-GB VM. Actual VM latency/RAM remain to be measured
  with the trained checkpoint and representative photos.

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
