from distutils.version import StrictVersion
import numpy as np
import os
import pandas as pd
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from tensorflow.python.platform import gfile

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops
from object_detection.nms import gpu_nms

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')
from utils import label_map_util
from utils import visualization_utils as vis_util

# NUM_CLASSES = 98
# PATH_TO_FROZEN_GRAPH = '/data5/xin/logo_jv/exported_models/frozen_inference_graph.pb'
# # PATH_TO_LABELS = '/home/rweber/models/research/object_detection/data/logo_jv1719.pbtxt'
# PATH_TO_LABELS = '/data5/xin/logo_jv/labels.pbtxt'
# OUTPUT_PATH = '/data5/xin/logo_jv/test_images_outputs2/'
# TEST_IMAGE_PATH = '/data5/xin/logo_jv/test_images/'
# # TEST_IMAGE_PATH = '/data5/xin/logo_jv/test_images_small/'
# NMS_THRESH=0.3

NUM_CLASSES = 57
PATH_TO_FROZEN_GRAPH = '/data5/xin/exported_models/irv2_atrous/frozen_inference_graph.pb'
PATH_TO_LABELS = '/data5/xin/irv2_atrous/labels.pbtxt'
OUTPUT_PATH = '/data5/xin/irv2_atrous/test_images_outputs_new/'
TEST_FILE = '/data5/xin/irv2_atrous/test.txt'
# TEST_FILE = '/data5/xin/irv2_atrous/test_small.txt'
# TEST_IMAGE_PATH = '/data1/xin/pipeline_brand_classifier/hive_test/data/'
NMS_THRESH=0.3
THRESH=0.01

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def run_nms(detection_boxes, detection_scores, detection_classes):
  combined = sorted(zip(detection_boxes, detection_scores, detection_classes), key=lambda x:x[2])
  combined_map = defaultdict(list)
  for box, score, classid in combined:
    combined_map[classid].append(list(box) + [score])
  # per class nms
  result_detection_boxes, result_detection_scores, result_detection_classes = [], [], []
  for key, vals in combined_map.iteritems():
    vals = np.array(vals).astype(np.float32)
    keep = gpu_nms.gpu_nms(vals, NMS_THRESH)
    vals = vals[keep, :]
    for val in vals:
      result_detection_boxes.append(val[:4])
      result_detection_scores.append(val[-1])
      result_detection_classes.append(int(key))
  return np.array(result_detection_boxes), np.array(result_detection_scores), np.array(result_detection_classes)

# def build_inference(image, graph):
def build_inference(graph):
  # Get handles to input and output tensors
  ops = tf.get_default_graph().get_operations()
  all_tensor_names = {output.name for op in ops for output in op.outputs}
  tensor_dict = {}
  for key in [
      'num_detections', 'detection_boxes', 'detection_scores',
      'detection_classes', 'detection_masks'
  ]:
    tensor_name = key + ':0'
    if tensor_name in all_tensor_names:
      tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
          tensor_name)
  image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
  return tensor_dict, image_tensor

if not os.path.exists(OUTPUT_PATH):
   os.makedirs(OUTPUT_PATH)
output_file = os.path.join(OUTPUT_PATH, 'output.csv')
output_file_with_thresh = os.path.join(OUTPUT_PATH, 'output_with_thresh.csv')

# search_path = os.path.join(TEST_IMAGE_PATH, '*')
# for image_path in gfile.Glob(search_path):

images = set()
with open(TEST_FILE, 'rb') as f:
  for line in f.readlines():
    image_path = line.strip().split(',')[0].strip()
    images.add(image_path)
images = list(images)

categories_dict = {}
for item in categories:
  categories_dict[item['id']] = item['name']

with detection_graph.as_default():
  with tf.Session() as sess:
    tensor_dict, image_tensor = build_inference(detection_graph)

    for image_path in images:

      image = Image.open(image_path)
      im_width, im_height = image.size
      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
      try:
        image_np = load_image_into_numpy_array(image)
      except:
        print('>>>>>>>>>>> Error: ', image_path)
        continue
      # # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      # image_np_expanded = np.expand_dims(image_np, axis=0)

      # Actual detection.
      # output_dict = run_inference_for_single_image(image_np, detection_graph)
      output_dict = sess.run(tensor_dict,
                         feed_dict={image_tensor: np.expand_dims(image_np, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]

      output_dict['detection_boxes'], output_dict['detection_scores'], output_dict['detection_classes'] = run_nms(
        output_dict['detection_boxes'],
        output_dict['detection_scores'],
        output_dict['detection_classes'])

      n = len(output_dict['detection_boxes'])
      print('>>>>>>>>>>>>>>>>>> n=', n)
      records = []
      records_with_thresh = []
      for idx in range(n):
        ymin, xmin, ymax, xmax = output_dict['detection_boxes'][idx]
        left, right, top, bottom = xmin*im_width, xmax*im_width, ymin*im_height, ymax*im_height

        score = output_dict['detection_scores'][idx]
        classname = categories_dict[output_dict['detection_classes'][idx]]
        
        records.append([image_path, 0, im_width, im_height, left, top, right, bottom, score, classname])
        if float(score) > THRESH:
          records_with_thresh.append([image_path, 0, im_width, im_height, left, top, right, bottom, score, classname])

      records_df = pd.DataFrame.from_records(records, columns=['path', 'timestamp', 'width', 'height', 'left', 'top', 'right', 'bottom', 'score', 'class'])
      records_df.to_csv(output_file, mode='a', index=False, header=False)

      records_df2 = pd.DataFrame.from_records(records_with_thresh, columns=['path', 'timestamp', 'width', 'height', 'left', 'top', 'right', 'bottom', 'score', 'class'])
      records_df2.to_csv(output_file_with_thresh, mode='a', index=False, header=False)

      # # Visualization of the results of a detection.
      # vis_util.visualize_boxes_and_labels_on_image_array(
      #     image_np,
      #     output_dict['detection_boxes'],
      #     output_dict['detection_classes'],
      #     output_dict['detection_scores'],
      #     category_index,
      #     instance_masks=output_dict.get('detection_masks'),
      #     use_normalized_coordinates=True,
      #     # line_thickness=8,
      #     min_score_thresh=THRESH)
      # # plt.figure(figsize=IMAGE_SIZE)
      # # plt.imshow(image_np)
      # output_image = os.path.join(OUTPUT_PATH, os.path.splitext(os.path.basename(image_path))[0]+'.png')
      # plt.imsave(output_image, image_np)

      print('>>>>>>>>>>>>>>> Done: ', image_path)
