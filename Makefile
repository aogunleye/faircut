.PHONY: install train predict monitor test dashboard

install:
	pip install -r requirements.txt

train:
	python src/train.py

predict:
	python src/predict.py

monitor:
	python src/monitoring.py

test:
	pytest tests/

dashboard:
	streamlit run app/dashboard.py