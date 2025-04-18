import os
import zipfile
import requests
from dotenv import load_dotenv

load_dotenv()

class ESC50Downloader:
    def __init__(self):
        self.url = os.getenv("ESC50_URL")
        self.data_root = os.getenv("DATA_ROOT")
        self.zip_path = os.path.join(self.data_root, "ESC-50.zip")
        self.extracted_path = os.path.join(self.data_root, "ESC-50-master")

    def dataset_exists(self):
        return os.path.exists(os.path.join(self.extracted_path, "audio")) and \
               os.path.exists(os.path.join(self.extracted_path, "meta/esc50.csv"))

    def download_dataset(self):
        if not os.path.exists(self.data_root):
            os.makedirs(self.data_root)

        print(" Скачивание ESC-50...")
        response = requests.get(self.url, stream=True)
        with open(self.zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)

    def extract_dataset(self):
        print(" Распаковка архива...")
        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.data_root)

    def prepare(self):
        if self.dataset_exists():
            print(" ESC-50 уже загружен и распакован.")
        else:
            self.download_dataset()
            self.extract_dataset()
            print(" Загрузка и распаковка завершены.")
