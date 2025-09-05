import numpy as np, time
from insightface.app import FaceAnalysis
from numpy.linalg import norm

class Recognizer:
  def __init__(self, providers=("CPUExecutionProvider",), det_size=(640,640)):
    self.app = FaceAnalysis(name="buffalo_l", providers=list(providers))
    self.app.prepare(ctx_id=0, det_size=det_size)

  @staticmethod
  def cosine_sim(a, b):
    return float(np.dot(a, b) / (norm(a) * norm(b) + 1e-9))

  def detect_embed(self, bgr):
    faces = self.app.get(bgr)
    # returns list of (bbox, embedding, kps)
    return [(f.bbox.astype(int), f.normed_embedding, f.kps) for f in faces]

  def match(self, emb, gallery, threshold=0.42):
    """
    emb: np.array of the detected face embedding
    gallery: list of (user_id, embedding_vector), embedding_vector can be None or empty
    threshold: cosine similarity threshold for a match
    """
    best_uid, best_sim = None, -1.0

    if emb is None or len(emb) == 0:
      # No embedding to match
      return (None, best_sim)

    for uid, g in gallery:
      if g is None or len(g) == 0:
        continue  # skip users without embeddings
      s = self.cosine_sim(emb, g)
      if s > best_sim:
        best_uid, best_sim = uid, s

    return (best_uid, best_sim) if best_sim >= threshold else (None, best_sim)

