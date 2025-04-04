import json
import os
import numpy as np
import pandas as pd


def text_send_ajax(CORE, NLP):
    print('PATH: /text_send_ajax/text_send_ajax.py')

    text = CORE.post['text']
    text = NLP.clear_text(text)
    df_text = pd.DataFrame([text])
    df_features = NLP.get_features(text)
    y_pred_v = NLP.model_rat.predict([df_text, df_features])
    y_pred_am = np.argmax(y_pred_v, axis=1)

    pred_list = NLP.le.inverse_transform(y_pred_am)
    rating = pred_list[0]
    
    answer = {'answer': 'success', 'rating': rating}
    return {'ajax': json.dumps(answer)}