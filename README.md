.# Deploying PyTorch models on Azure ML using TorchServe

An example of how to package PyTorch models with TorchServe and deploy them using Azure ML. This example combines and closely follows the official [PyTorch TorchServe example](https://github.com/pytorch/serve/tree/master/examples/image_classifier/resnet_18) and Azure ML TorchServe example ([blog post](https://techcommunity.microsoft.com/t5/ai-machine-learning-blog/deploy-pytorch-models-with-torchserve-in-azure-machine-learning/ba-p/2466459), [repository](https://github.com/Azure/azureml-examples/blob/main/cli/deploy-custom-container-torchserve-densenet.sh), [additional repository](https://github.com/Azure/azureml-examples/tree/main/cli/endpoints/online/custom-container/torchserve/densenet)).

## Walkthrough

### 1. Install the reqirements

This example was written and tested using Python 3.9.19, PyTorch 2.3.1, Torchvision 0.18.1, and Torch Model Archiver
0.11.0 on macOS 14.5 (Apple Silicon). For a full list of requirements see `requirements.txt`.

Ideally after creating an isolated virtual environment (e.g., using `virtualenv`, `pyenv`, or `conda`), install the
requirements by executing the following.

```bash
pip install -r requirements.txt
```

### 2. Download the model checkpoint

For this example, we'll use the official Torchvision ResNet-18 ImageNet model weights. These can be downloaded as
follows.

```bash
curl --output resnet18-f37072fd.pth https://download.pytorch.org/models/resnet18-f37072fd.pth
```

### 3. Export the model as a "Model Archive Repository" (.mar) file

```bash
mkdir -p torchserve
torch-model-archiver \
    --model-name resnet18 \
    --version 1.0 \
    --model-file ./model.py \
    --serialized-file ./resnet18-f37072fd.pth \
    --export-path ./torchserve \
    --extra-files ./index_to_name.json \
    --handler ./handler.py\
    --config ./config.yaml
```

### 4. Download a test image

```bash
curl -O https://raw.githubusercontent.com/pytorch/serve/master/docs/images/kitten_small.jpg
```

### 5. Perform a basic test

#### 5.1. Start TorchServe to serve the model

In another terminal session, start TorchServe.

```bash
torchserve --start --ncs --model-store torchserve --models resnet18.mar
```

#### 5.2. Generate a prediction

```bash
curl http://127.0.0.1:8080/predictions/resnet18 -T kitten_small.jpg
```

#### 5.3. Stop TorchServe

```bash
torchserve --stop
```

### 6. Deploy on Azure

#### 6.1. Build the image

For this step you'll need to have installed and configured the [Azure CLI]() and the [`ml` extension](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-configure-cli?view=azureml-api-2&tabs=public). You'll also need to have installed and running Docker Desktop.

```bash
BASE_PATH=.
ENDPOINT_NAME=endpt-torchserve-`echo $RANDOM`

# Get name of workspace and Azure Container Registry (ACR)
WORKSPACE=$(az config get --query "defaults[?name == 'workspace'].value" -o tsv)
ACR_NAME=$(az ml workspace show --name $WORKSPACE --query container_registry -o tsv | cut -d'/' -f9-)

# Build image
IMAGE_TAG=${ACR_NAME}.azurecr.io/torchserve:1
docker build --platform linux/amd64 -t $IMAGE_TAG $BASE_PATH
docker push $IMAGE_TAG
```

#### 6.2. Test the model locally in a containerized environment

```bash
# Run image locally for testing
docker run --rm -d -p 8080:8080 --platform linux/amd64 --name torchserve-test \
  -e AZUREML_MODEL_DIR=/var/azureml-app/azureml-models/ \
  -e TORCHSERVE_MODELS="resnet18=resnet18.mar" \
  -v $PWD/$BASE_PATH/torchserve:/var/azureml-app/azureml-models/torchserve $IMAGE_TAG

# Check Torchserve health
echo "Checking Torchserve health..."
curl http://localhost:8080/ping

# Check scoring locally
echo "Uploading testing image, the scoring is..."
curl http://localhost:8080/predictions/densenet161 -T kitten_small.jpg

# Stop container
docker stop torchserve-test
```

## Notes

- You may need to enable the ACR admin user under Settings > Access keys to push images to the container registry.
- Reduced instance type from Standard_DS3_v2 to Standard_DS2_v2 due to free tier limits.