# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path, Secret
import os
import time
import subprocess
from zipfile import ZipFile

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        print("Starting setup")
        t1 = time.time()
        # Cleanup previous runs
        os.system("rm -rf cache")
        os.system("rm -rf datasets")
        os.system("rm -rf output")
        # Download FLUX weights
        t2 = time.time()
        print(f"Setup time: {t2-t1} seconds")

    def predict(
        self,
        images: Path = Input(
            description="A .zip or .tar file containing the image files that will be used for fine-tuning. Image names must be their captions: watercolor_tiger.png, etc. Minimum 12 images required."
        ),
        hf_token: Secret = Input(
            description="HuggingFace token to use for accessing the FLUX-Dev weights"
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        print("Starting prediction")
        # Set huggingface token via huggingface-cli login
        os.system(f"huggingface-cli login --token {hf_token.get_secret_value()}")

        # Unzip images from input images file to the dataset/replicate folder
        datasets_dir = "datasets/Replicate"
        input_images = str(images)
        if input_images.endswith(".zip"):
            print("Detected zip file")
            os.makedirs("datasets", exist_ok=True)
            with ZipFile(input_images, "r") as zip_ref:
                zip_ref.extractall(datasets_dir+"/")
            print("Extracted zip file")
        elif input_images.endswith(".tar"):
            print("Detected tar file")
            os.makedirs(datasets_dir, exist_ok=True)
            os.system(f"tar -xvf {input_images} -C {datasets_dir}")
            print("Extracted tar file")

        # Run - bash train.sh
        subprocess.run(["git", "config", "--global", "--add", "safe.directory", "/src"])
        subprocess.check_call(["bash", "train.sh"], close_fds=False)

        # Zip up the output folder
        os.system("zip -r /tmp/output.zip output")
        return Path("/tmp/output.zip")
