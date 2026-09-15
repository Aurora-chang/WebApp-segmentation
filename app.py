"""Single-process Flask application; the model is constructed once by create_app."""
import argparse
import base64
from io import BytesIO
import logging
import os
from threading import Lock

from flask import Flask, render_template, request
import torch
from waitress import serve
from werkzeug.exceptions import BadRequest, RequestEntityTooLarge

from inference import InvalidImage, Segmenter

MAX_UPLOAD_BYTES = 10 * 1024 * 1024


def image_url(image, format):
    buffer = BytesIO()
    image.save(buffer, format=format)
    mime = "image/jpeg" if format == "JPEG" else "image/png"
    return f"data:{mime};base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def create_app(checkpoint_path=None):
    app = Flask(__name__)
    app.config.update(MAX_CONTENT_LENGTH=MAX_UPLOAD_BYTES, MAX_FORM_PARTS=4)
    path = checkpoint_path or os.environ.get("PET_CHECKPOINT")
    if not path:
        raise ValueError("Set PET_CHECKPOINT to the full path of your trained best.pt.")
    torch.set_num_threads(2)
    segmenter = Segmenter(path)
    app.extensions["segmenter"] = segmenter
    busy = Lock()
    app.extensions["prediction_lock"] = busy

    def error(message, status):
        return render_template("index.html", error=message), status

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.post("/segment")
    def segment():
        # Bound memory for decoding, inference, and result encoding together.
        if not busy.acquire(blocking=False):
            return error("Another photo is being processed. Please try again shortly.", 503)
        try:
            upload = request.files.get("image")
            if upload is None or not upload.filename:
                return error("Please choose an image to upload.", 400)
            result = segmenter.predict(upload.stream)
            return render_template(
                "index.html",
                original=image_url(result.original, "JPEG"),
                overlay=image_url(result.overlay, "JPEG"),
                mask=image_url(result.mask, "PNG"),
            )
        except InvalidImage as exc:
            return error(str(exc), 400)
        except (RequestEntityTooLarge, BadRequest):
            raise
        except Exception:
            app.logger.exception("Photo segmentation failed")
            return error("The photo could not be processed. Please try again.", 500)
        finally:
            busy.release()

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_):
        return error("Upload is too large. Keep the entire upload below 10 MiB.", 413)

    @app.errorhandler(BadRequest)
    def bad_request(_):
        return error("The upload could not be read. Please select your image again.", 400)

    @app.after_request
    def no_cache(response):
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", help="Overrides PET_CHECKPOINT")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    try:
        app = create_app(args.checkpoint)
    except ValueError as exc:
        parser.error(str(exc))
    # Two HTTP threads, one model instance, at most one prediction at a time.
    serve(app, host=args.host, port=args.port, threads=2,
          max_request_body_size=MAX_UPLOAD_BYTES, connection_limit=16)


if __name__ == "__main__":
    main()
