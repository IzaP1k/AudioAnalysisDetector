# AudioAnalysisDetector

Krótki opis projektu: detekcja fałszywych/przekształconych nagrań audio z wykorzystaniem klasycznych metod ML i sieci głębokich (ResNet, BiLSTM).

## Struktura folderów
- `ASV_dataset.ipynb`, `GMM_BiLSTM_notebook.ipynb`, `ResNet_notebook.ipynb`, `ml_model_gridsearch_notebook.ipynb`, `xai_notebook.ipynb`, `data_augmentation.ipynb`: notatniki pokazujące procesy przygotowania danych, treningu i analizy.
- `run_all_task.py`: główny pipeline do przygotowania danych i treningu modeli.
- `prepare_data.py`, `ASV_dl_func.py`, `classic_ml_func.py`, `pytorch_func.py`, `visualisation_func.py`, `xai_func.py`, `gmm_bilstm_pipeline.py`: moduły z funkcjami do przetwarzania danych, ekstrakcji cech, treningu i ewaluacji.
- `config.yaml`: konfiguracja ścieżek i zbiorów danych.
- `Res_Net/`: katalog z wynikami treningów modeli ResNet (logi, metryki, checkpointy).
- `GMM-BiLSTM/`: katalog z wynikami treningów modeli GMM + BiLSTM (logi, metryki, checkpointy).
- `csvki/`: pomocnicze pliki `.csv` z przygotowanymi/przefiltrowanymi danymi, aby minimalizować wymagane zasoby obliczeniowe podczas kolejnych uruchomień.
- `best_model/`: najlepsze wytrenowane modele i powiązane standaryzacje (np. pliki `.pt`, skaler), gotowe do użycia.

## Wymagania i instalacja
1. Utwórz wirtualne środowisko (PowerShell):
```powershell
python -m venv venv
& .\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

## Dane i pobieranie
- In-the-wild: https://zenodo.org/records/7874857
- ASVspoof 2019: https://www.asvspoof.org/

Po pobraniu ustaw ścieżki w `config.yaml` lub przez zmienne środowiskowe (`ASVSPOOF_META`, `ASVSPOOF_FLAC`).

### Konfiguracja `config.yaml`
Przykładowa, czysta struktura (uzupełnij własnymi ścieżkami lub zmiennymi środowiskowymi):
```yaml
paths:
  metadata_path: ${env:ASVSPOOF_META}
  flac_folder: ${env:ASVSPOOF_FLAC}
  in_the_wild_dir: "C:/data/in_the_wild"

datasets:
  LA:
    metadata: "C:/data/asvspoof2019/LA/metadata.txt"
    flac:
      - "C:/data/asvspoof2019/LA/flac"
    columns:
      - speaker_id
      - file_id
      - codec
      - corpus
      - attack_id
      - label
      - trim
      - set

```

## Jak uruchomić i co znaleźć w notatnikach
- `ASV_dataset.ipynb`: przygotowanie i eksploracja danych ASV.
- `data_augmentation.ipynb`: przykłady augmentacji audio.
- `GMM_BiLSTM_notebook.ipynb`: pipeline ekstrakcji CQCC i trening BiLSTM.
- `ResNet_notebook.ipynb`: trening modeli ResNet na wybranych cechach/spektrogramach.
- `ml_model_gridsearch_notebook.ipynb`: siatka hiperparametrów dla klasycznych modeli ML.
- `xai_notebook.ipynb`: interpretowalność modeli (XAI).

## Skrót funkcjonalności plików `.py`
- `prepare_data.py`: budowa dataframe, łączenie zbiorów, balansowanie klas, ekstrakcja cech (CQCC, GTCC, mel-spectrogram, MFCC, LFCC).
- `ASV_dl_func.py`: augmentacja, ekstrakcja cech, przygotowanie zbiorów, trening wszystkich cech, funkcje pomocnicze.
- `classic_ml_func.py`: grid search, trening i ewaluacja klasycznych modeli.
- `pytorch_func.py`: wspólne funkcje dla modeli PyTorch.
- `gmm_bilstm_pipeline.py`: BiLSTM, DataLoader, ewaluacja modeli GMM-BiLSTM.
- `visualisation_func.py`: wykresy i wizualizacje metryk.
- `xai_func.py`: metody XAI dla modeli.

## Szybki start
- Sklonuj repozytorium:
```powershell
git clone https://github.com/IzaP1k/AudioAnalysisDetector.git
cd AudioAnalysisDetector
python -m venv venv
& .\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```
- Skonfiguruj `config.yaml`.
- Uruchom trening/pipeline: `python run_all_task.py`.
- Wyniki pojawią się w `Res_Net/` oraz `GMM-BiLSTM/`, a pomocnicze dane w `csvki/`.
