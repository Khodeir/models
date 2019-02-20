import sys
sys.path.append('/Users/mohamedkhodeir/Documents/models/research')
sys.path.append('/Users/mohamedkhodeir/Documents/models/research/slim')
from google.protobuf import text_format
from object_detection.builders import box_predictor_builder
from object_detection.protos import box_predictor_pb2
import IPython
box_predictor_text_proto = '''keypoint_predictor {
    min_depth: 0
    max_depth: 0
    num_layers_before_predictor: 0
    use_dropout: false
    dropout_keep_probability: 0.8
    kernel_size: 1
    box_code_size: 4
    apply_sigmoid_to_scores: false
    conv_hyperparams {
      activation: RELU_6,
      regularizer {
        l2_regularizer {
          weight: 0.00004
        }
      }
      initializer {
        truncated_normal_initializer {
          stddev: 0.03
          mean: 0.0
        }
      }
      batch_norm {
        train: true,
        scale: true,
        center: true,
        decay: 0.9997,
        epsilon: 0.001,
      }
    }
}'''
box_predictor_proto = box_predictor_pb2.BoxPredictor()
text_format.Merge(box_predictor_text_proto, box_predictor_proto)

box_predictor_builder.build(argscope_fn=lambda x, y: 'Stuff', box_predictor_config=box_predictor_proto, is_training=True, num_classes=1)