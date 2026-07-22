import os
import random
from locust import HttpUser, task, between

SAMPLE_IMAGES_DIR = "locust_sample_images"


class BookGenreClassifierUser(HttpUser):
    wait_time = between(1, 3)

    def on_start(self):
        if os.path.exists(SAMPLE_IMAGES_DIR):
            self.sample_files = [
                os.path.join(SAMPLE_IMAGES_DIR, f)
                for f in os.listdir(SAMPLE_IMAGES_DIR)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
        else:
            self.sample_files = []

    @task(3)
    def predict(self):
        if not self.sample_files:
            return
        image_path = random.choice(self.sample_files)
        with open(image_path, "rb") as f:
            self.client.post(
                "/predict",
                files={"file": (os.path.basename(image_path), f, "image/jpeg")},
                name="/predict"
            )

    @task(1)
    def health_check(self):
        self.client.get("/health", name="/health")