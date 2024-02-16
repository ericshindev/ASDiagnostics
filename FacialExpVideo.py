## venv : py38openCV

from deepface import DeepFace
import numpy as np

import cv2
vidcap = cv2.VideoCapture('video/5.mp4')

assert vidcap.isOpened()

fps_in = vidcap.get(cv2.CAP_PROP_FPS)
fps_out = 3

index_in = -1
index_out = -1

AnalyResult = []
while True:
    success = vidcap.grab()
    if not success: break
    index_in += 1

    out_due = int(index_in / fps_in * fps_out)
    if out_due > index_out:
        success, frame = vidcap.retrieve()
        if not success: break
        index_out += 1

        # do something with `frame`
        try:
          predictions = DeepFace.analyze(frame)

          print(predictions)
          AnalyResult.append(predictions)
        except:
          pass


import pickle


# save
with open('5_faceAnalysisData.pickle', 'wb') as f:
    pickle.dump(AnalyResult, f, pickle.HIGHEST_PROTOCOL)

# load
# with open('faceAnalysisData.pickle', 'rb') as f:
#     data = pickle.load(f)

# numpy, pandas + 기타 분석 방법,,주로는 NLP 기술을 활용하여 선수선발에 활용할 수 있는 인공지능 + 알고리즘을 만들어 낼 것이다.