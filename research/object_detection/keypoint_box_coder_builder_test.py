import sys
sys.path.append('/Users/mohamedkhodeir/Documents/models/research')
sys.path.append('/Users/mohamedkhodeir/Documents/models/research/slim')
from google.protobuf import text_format
from object_detection.builders import box_coder_builder
from object_detection.protos import box_coder_pb2
import IPython
box_coder_text_proto = '''keypoint_box_coder {
  num_keypoints: 4
  y_scale: 10.0
  x_scale: 10.0
  height_scale: 5.0
  width_scale: 5.0
}'''
box_coder_proto = box_coder_pb2.BoxCoder()
text_format.Merge(box_coder_text_proto, box_coder_proto)

box_coder_builder.build(box_coder_proto)