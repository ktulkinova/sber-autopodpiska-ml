import pandas as pd
from catboost import CatBoostClassifier


num_features = ['visit_number', 'visit_hour', 'visit_dow']
cat_features = ['utm_source', 'utm_medium', 'utm_campaign', 'utm_adcontent',
                'device_category', 'device_os', 'device_brand', 'device_browser',
                'geo_country', 'geo_city']
extra_features = ['is_organic']
features = num_features + cat_features + extra_features


model = CatBoostClassifier()
model.load_model('catboost_model.cbm')


def predict_visit(visit: dict) -> int:
    
    df = pd.DataFrame([visit])

    organic_mediums = ['organic', 'referral', '(none)']
    df['is_organic'] = df['utm_medium'].isin(organic_mediums).astype(int)
    df[cat_features] = df[cat_features].astype(str)
    prediction = model.predict(df[features])[0]
    return int(prediction)

if __name__ == '__main__':
    example_visit = {
        'visit_number': 1,
        'visit_hour': 14,
        'visit_dow': 2,
        'utm_source': 'ZpYIoDJMcFzVoPFsHGJL',
        'utm_medium': 'banner',
        'utm_campaign': 'LEoPHuyFvzoNfnzGgfcd',
        'utm_adcontent': 'vCIpmpaGBnIQhyYNkXqp',
        'device_category': 'mobile',
        'device_os': 'Android',
        'device_brand': 'Huawei',
        'device_browser': 'Chrome',
        'geo_country': 'Russia',
        'geo_city': 'Moscow'
    }
    result = predict_visit(example_visit)
    print('Предсказание (0/1):', result)