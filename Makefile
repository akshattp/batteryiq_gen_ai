.PHONY: setup data train test app report help

help:
	@echo "Available commands:"
	@echo "  make setup   - Install project dependencies"
	@echo "  make data    - Ingest, validate, clean raw NASA battery data"
	@echo "  make train   - Train VAE anomaly detector and temperature predictor"
	@echo "  make test    - Run test suite via pytest"
	@echo "  make app     - Launch Streamlit interactive dashboard"
	@echo "  make report  - Generate Model Card and Auto-Heal reports"

setup:
	python -m pip install -r requirements.txt

data:
	python scripts/build_dataset.py

train:
	python scripts/train_vae.py
	python scripts/train_temp.py

test:
	pytest tests/ -v

app:
	streamlit run app/Home.py

report:
	python tools/make_model_card.py
	python tools/make_autoheal_report.py

colab-pack:
	python scripts/make_colab_pack.py
