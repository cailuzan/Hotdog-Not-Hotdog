import tensorflow.compat.v1 as tf
import os
import sys

tf.disable_v2_behavior()  # 禁用TensorFlow 2.x行为，启用1.x兼容模式

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

input_layer = 'DecodeJpeg/contents:0'
output_layer = 'final_result:0'

# 验证文件路径
if len(sys.argv) < 2:
    print("Please provide the image file path as an argument.")
    sys.exit(1)

image_path = sys.argv[1]

if not tf.io.gfile.exists(image_path):
    print(f"File does not exist: {image_path}")
    sys.exit(1)

print(f'Reading image from: {image_path}')

# 读取图像数据
with tf.io.gfile.GFile(image_path, 'rb') as f:
    image = f.read()

# 读取标签文件
labels = [line.rstrip() for line in tf.io.gfile.GFile("labels_hotdog.txt", 'r')]
with tf.io.gfile.GFile("graph_hotdog.pb", 'rb') as f:
    g_def = tf.GraphDef()
    g_def.ParseFromString(f.read())
    tf.import_graph_def(g_def, name='')

with tf.Session() as sess:
    soft_tensor = sess.graph.get_tensor_by_name(output_layer)
    predict, = sess.run(soft_tensor, {input_layer: image})
    top = predict.argsort()[-2:][::-1]
    for x in top:
        result = labels[x]
        score = predict[x]
        print('%s  =  %.5f ' % (result, score))
