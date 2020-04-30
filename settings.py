import os


CUR_DIR = os.path.dirname(os.path.abspath(__file__))

FRONT_FACE_DETECTION_MODEL = os.path.join(CUR_DIR, 'utils', 'model', 'haarcascade_frontalface_default.xml')
MASK_MODEL_PATH = os.path.join(CUR_DIR, 'utils', 'model', 'frozen_inference_graph.pb')
LIVE_MODEL_PATH = os.path.join(CUR_DIR, 'utils', 'model', 'liveness.model')
LIVE_LE_PATH = os.path.join(CUR_DIR, 'utils', 'model', 'le.pickle')

THRESHOLD = 0.9
