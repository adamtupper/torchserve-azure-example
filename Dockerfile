FROM pytorch/torchserve:latest-cpu

CMD ["torchserve","--start","--model-store","$AZUREML_MODEL_DIR/torchserve","--models","resnet18.mar"]