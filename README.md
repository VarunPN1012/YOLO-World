# Pytorch to ONNX conversion

Link to download the model - https://huggingface.co/wondervictor/YOLO-World/blob/main/yolo_world_v2_s_obj365v1_goldg_pretrain-55b943ea.pth

To pull the docker base image and create another image with all the required dependencies for setting up environment for .pt to onnx conversion do run 

```
docker build -t docker_image_name/with/required/dependencies -f Dockerfile .
```
Source for lvis_v1_minival_inserted_image_name.json,  https://huggingface.co/GLIPModel/GLIP/blob/main/lvis_v1_minival_inserted_image_name.json

```
docker run -it --gpus all --name pytorch_onnx_conversion_image -v project/host/directory:/workspace docker_image_name/with/required/dependencies
```

To Run the pytorch to onnx conversion 


For setting the prompt based classes for the model just modify the given ``` labels.json ```

```
PYTHONPATH=./ python deploy/export_onnx.py configs/pretrain/yolo_world_v2_s_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py path/to/downloaded/pytorch_model --custom-text path/to/labels.json --opset 12 --without-nms --backend tensorrt8
```

# YOLO-World TensorRT conversion and Inference

Provides a high-performance C++ implementation for running YOLO-World ONNX models using NVIDIA TensorRT. It is optimized for deployment in production environments requiring fast and efficient object detection.

## Dependencies

Ensure the following dependencies are installed before building the project:

- **CUDA**: 12.2  
- **TensorRT**: 8.6.1  
- **OpenCV** : 4.8.0

(The conversion will work on CUDA 12.6 + TensorRT 10.3.1 environment as well , Tested)

## Build Instructions

Inside ``` triton_tool ``` directory 

To build the project, follow these steps:

```bash
# Navigate to the main project directory
mkdir build && cd build

# Configure the CMake project
cmake ..

# Build with 4 parallel jobs
make -j4

```
## For running the inference build 

To save the inferene results for verifying the created TensorRT model , create a directory ``` results ```


```bash
./yolo-world /path/to/onnx_file /path/to/dataset
```

sample dataset link : https://drive.google.com/drive/folders/1gTP-hTk2rBIxutOQ2EVeeRPKzYchjiOn?usp=sharing


# TritonServer setup 

Inside the ```yolo_world_triton_setup```

The model_repository for yolo world model is 

```
model_repository/
|── yolo_world/
    ├── 1/
    │   └── model.plan
    └── config.pbtxt
```

Rename the generated engine file with ``` model.plan ``` and add to the respective model_repository structure 

Triton server configuration --> ```config.pbtxt```
```
name: "yolo_world"
platform: "tensorrt_plan"
max_batch_size: 1  

input [
  {
    name: "images"
    data_type: TYPE_FP32
    dims: [3, 640, 640]  # CHW format (without batch dimension)
    format: FORMAT_NCHW  # Explicitly specify NCHW format
  }
]

output [
  {
    name: "boxes"
    data_type: TYPE_FP32
    dims: [8400, 4]  # Remove batch dimension from config
  },
  {
    name: "scores"
    data_type: TYPE_FP32
    dims:[8400, 3]  # Change according to the number of classes --> [8400,num_classes]
  }
]

instance_group [
  {
    kind: KIND_GPU
    count: 1
  }
]

```

To up the Triton server add model_repository path to the ``` docker_run.sh ``` bash script 

To Run the Triton server

make the bash script to executable 
``` 
chmod +x docker_run.sh 
```

Then Run executable 

```
 ./docker_run.sh
```


