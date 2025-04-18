import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
from tqdm import tqdm
from PIL import Image
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
import shutil

load_dotenv()

class ESC50Preprocessor:
    def __init__(self):
        self.data_root = os.getenv("DATA_ROOT")
        self.audio_dir = os.path.join(self.data_root, "ESC-50-master/audio")
        self.meta_csv = os.path.join(self.data_root, "ESC-50-master/meta/esc50.csv")
        self.sample_rate = int(os.getenv("SAMPLE_RATE"))
        self.spec_size = (int(os.getenv("SPEC_WIDTH")), int(os.getenv("SPEC_HEIGHT")))
        self.val_size = float(os.getenv("VAL_SIZE"))
        self.test_size = float(os.getenv("TEST_SIZE"))
        self.output_dir = os.path.join(self.data_root, "processed")

    def preprocess(self):
        df = pd.read_csv(self.meta_csv)
        X, y = [], []

        for _, row in tqdm(df.iterrows(), total=len(df)):
            filename = row["filename"]
            label = row["category"]
            path = os.path.join(self.audio_dir, filename)

            try:
                y_audio, sr = librosa.load(path, sr=self.sample_rate)
                S = librosa.stft(y_audio)
                S_db = librosa.amplitude_to_db(abs(S))

                # Resize
                S_img = Image.fromarray(S_db)
                S_img = S_img.resize(self.spec_size)
                S_arr = np.array(S_img)

                # Normalize
                S_norm = (S_arr - S_arr.min()) / (S_arr.max() - S_arr.min())

                X.append(S_norm)
                y.append((filename, label))

            except Exception as e:
                print(f"⚠️ Ошибка обработки {filename}: {e}")

        return np.array(X), y

    def split_and_save(self):
        print(" Обработка и сохранение...")
        X, y_data = self.preprocess()
        filenames, labels = zip(*y_data)

        df = pd.DataFrame({'filename': filenames, 'label': labels})
        X_train, X_temp, y_train, y_temp = train_test_split(X, df, test_size=self.val_size + self.test_size, stratify=labels, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=self.test_size / (self.val_size + self.test_size), stratify=y_temp['label'], random_state=42)

        for subset_name, X_set, y_set in zip(['train', 'val', 'test'], [X_train, X_val, X_test], [y_train, y_val, y_test]):
            subset_path = os.path.join(self.output_dir, subset_name)
            if os.path.exists(subset_path):
                shutil.rmtree(subset_path)
            os.makedirs(subset_path)

            for i, (spec, row) in enumerate(zip(X_set, y_set.iterrows())):
                _, row_data = row
                label = row_data['label']
                label_path = os.path.join(subset_path, label)
                os.makedirs(label_path, exist_ok=True)
                np.save(os.path.join(label_path, f"{row_data['filename'].replace('.wav', '')}.npy"), spec)

        print(" Спектрограммы сохранены в", self.output_dir)
