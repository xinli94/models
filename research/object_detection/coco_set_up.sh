cd /workspace/xli/models/research
protoc object_detection/protos/*.proto --python_out=.
rm -rf cocoapi/
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make
cp -r pycocotools /workspace/xli/models/research/
cd /workspace/xli/models/research
python object_detection/builders/model_builder_test.py