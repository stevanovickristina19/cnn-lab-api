# Build Your Own CNN 

## About

This repository provides an API for creating custom Convolutional Neural Networks (CNNs).
You can define the number of layers and configure all key parameters, experiment with different architectures, 
train your model, and evaluate the results to find the best-performing configuration


## Installation

Clone this repository:

```bash 
git clone https://github.com/stevanovickristina19/cnn-lab-api.git
```

Make virtual environment and install requirements
```bash 
cd cnn-lab-api && \
python3 -m venv cnnlab && \
source cnnlab/bin/activate && \
pip3 install requirements.txt
```

## Prepare dataset
```bash
mkdir -p data && \
wget -O data/flower_photos.tgz https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz && \
tar -xvzf data/flower_photos.tgz -C data && \
mv data/flower_photos data/flowers && \
rm data/flower_photos.tgz
```

## Run the server

To run the server:
```bash
python3 -m uvicorn app.main:app --reload
```

# Functionallities

- Load models
- Create model with predefined or custom architecture 
- Train created model
- Get training plots
- Test model on specific image

