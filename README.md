# Pytorch to ONNX conversion

Link to download the model - https://huggingface.co/wondervictor/YOLO-World/blob/main/yolo_world_v2_s_obj365v1_goldg_pretrain-55b943ea.pth

To pull the docker base image and create another image with all the required dependencies for setting up environment for .pt to onnx conversion do run

On the root directory 

```
docker build -t docker_image_name/with/required/dependencies -f Dockerfile .
```
Source for lvis_v1_minival_inserted_image_name.json,  https://huggingface.co/GLIPModel/GLIP/blob/main/lvis_v1_minival_inserted_image_name.json
(Already it is there in the `data/coco`)
```
docker run -it --gpus all --name pytorch_onnx_conversion_image -v project/host/directory:/workspace docker_image_name/with/required/dependencies
```

To Run the pytorch to onnx conversion 

For setting the prompt based classes for the model just modify the given ``` labels.json ```

```
PYTHONPATH=./ python deploy/export_onnx.py configs/pretrain/yolo_world_v2_s_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py path/to/downloaded/pytorch_model --custom-text path/to/labels.json --opset 12 --without-nms --backend tensorrt8
```

And the onnx model will be saved in the `work_dirs` in the root folder