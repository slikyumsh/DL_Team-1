from downloader import ESC50Downloader
from preprocessor import ESC50Preprocessor

if __name__ == "__main__":
    print(" Проверка и загрузка датасета...")
    downloader = ESC50Downloader()
    downloader.prepare()

    print("\n Предобработка...")
    preprocessor = ESC50Preprocessor()
    preprocessor.split_and_save()
