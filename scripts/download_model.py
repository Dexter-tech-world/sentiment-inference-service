"""Download the model at image build time so containers run fully offline.

Uses from_pretrained (not snapshot_download) so exactly the files the runtime
loads are fetched -- the repo also holds ONNX/TF/Rust weights we don't need.
"""

import os

from transformers import AutoModelForSequenceClassification, AutoTokenizer

name = os.environ["MODEL_NAME"]
revision = os.environ["MODEL_REVISION"]
AutoTokenizer.from_pretrained(name, revision=revision)
AutoModelForSequenceClassification.from_pretrained(name, revision=revision)
print(f"cached {name}@{revision}")
