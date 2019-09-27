import json

import numpy as np


class Modeller:

    def __init__(self, model_class, settings=None):
        if not settings:
            settings = {}

        if isinstance(settings, str):
            with open(settings, "r") as fileobj:
                settings = json.load(fileobj)

        self._model = model_class(**settings)

    def fit(self, data):
        self._fitted_model = self._model.fit(data.x, data.y)
        self._pred_x = self._fitted_model.predict(data.x)
        print(f"The accuracy is {np.round(np.mean(self._pred_x==data.y), 2)}")
