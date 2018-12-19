import functools
import os
import tensorflow as tf

from object_detection.builders import dataset_builder
from object_detection.builders import graph_rewriter_builder
from object_detection.builders import model_builder
from object_detection.legacy import evaluator
from object_detection.utils import config_util
from object_detection.utils import label_map_util

# init
configs = config_util.get_configs_from_pipeline_file(
    FLAGS.pipeline_config_path)
tf.gfile.Copy(
    FLAGS.pipeline_config_path,
    os.path.join(FLAGS.eval_dir, 'pipeline.config'),
    overwrite=True)

model_config = configs['model']
eval_config = configs['eval_config']
input_config = configs['eval_input_config']

model_fn = functools.partial(
  model_builder.build, model_config=model_config, is_training=False)

def get_next(config):
  return dataset_builder.make_initializable_iterator(
      dataset_builder.build(config)).get_next()

create_input_dict_fn = functools.partial(get_next, input_config)

categories = label_map_util.create_categories_from_labelmap(
    input_config.label_map_path)


def _extract_predictions_and_losses(model,
                                    create_input_dict_fn,
                                    ignore_groundtruth=False):
  input_dict = create_input_dict_fn()
  prefetch_queue = prefetcher.prefetch(input_dict, capacity=500)
  input_dict = prefetch_queue.dequeue()
  original_image = tf.expand_dims(input_dict[fields.InputDataFields.image], 0)
  preprocessed_image, true_image_shapes = model.preprocess(
      tf.to_float(original_image))
  prediction_dict = model.predict(preprocessed_image, true_image_shapes)
  detections = model.postprocess(prediction_dict, true_image_shapes)

  groundtruth = None
  losses_dict = {}
  if not ignore_groundtruth:
    groundtruth = {
        fields.InputDataFields.groundtruth_boxes:
            input_dict[fields.InputDataFields.groundtruth_boxes],
        fields.InputDataFields.groundtruth_classes:
            input_dict[fields.InputDataFields.groundtruth_classes],
        fields.InputDataFields.groundtruth_area:
            input_dict[fields.InputDataFields.groundtruth_area],
        fields.InputDataFields.groundtruth_is_crowd:
            input_dict[fields.InputDataFields.groundtruth_is_crowd],
        fields.InputDataFields.groundtruth_difficult:
            input_dict[fields.InputDataFields.groundtruth_difficult]
    }
    # if fields.InputDataFields.groundtruth_group_of in input_dict:
    #   groundtruth[fields.InputDataFields.groundtruth_group_of] = (
    #       input_dict[fields.InputDataFields.groundtruth_group_of])
    # groundtruth_masks_list = None
    # if fields.DetectionResultFields.detection_masks in detections:
    #   groundtruth[fields.InputDataFields.groundtruth_instance_masks] = (
    #       input_dict[fields.InputDataFields.groundtruth_instance_masks])
    #   groundtruth_masks_list = [
    #       input_dict[fields.InputDataFields.groundtruth_instance_masks]]
    # groundtruth_keypoints_list = None
    # if fields.DetectionResultFields.detection_keypoints in detections:
    #   groundtruth[fields.InputDataFields.groundtruth_keypoints] = (
    #       input_dict[fields.InputDataFields.groundtruth_keypoints])
    #   groundtruth_keypoints_list = [
    #       input_dict[fields.InputDataFields.groundtruth_keypoints]]
    label_id_offset = 1
    model.provide_groundtruth(
        [input_dict[fields.InputDataFields.groundtruth_boxes]],
        [tf.one_hot(input_dict[fields.InputDataFields.groundtruth_classes]
                    - label_id_offset, depth=model.num_classes)],
        groundtruth_masks_list, groundtruth_keypoints_list)
    losses_dict.update(model.loss(prediction_dict, true_image_shapes))

  result_dict = eval_util.result_dict_for_single_example(
      original_image,
      input_dict[fields.InputDataFields.source_id],
      # input_dict[fields.InputDataFields.filename],
      detections,
      groundtruth,
      class_agnostic=(
          fields.DetectionResultFields.detection_classes not in detections),
      scale_to_absolute=True)
  return result_dict, losses_dict

# run
model = model_fn()