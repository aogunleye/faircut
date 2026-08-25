import pandas as pd
from src.features import FeatureEngineeringPipeline

def test_features_transformation():
    raw_df = pd.DataFrame([{
        "id": 1,
        "title": "Test Movie",
        "popularity": 20.0,
        "original_language": "en",
        "budget": 10000000,
        "revenue": 50000000,
        "vote_average": 7.5,
        "vote_count": 100,
        "runtime": 120,
        "belongs_to_collection": 1,
        "genres": ["Action", "Drama"],
        "keywords": ["hero", "space"],
        "certification": "PG-13",
        "production_companies": ["Universal Pictures"],
        "director_gender": 2
    }])
    
    pipeline = FeatureEngineeringPipeline(top_k_keywords=10)
    processed_df = pipeline.fit_transform(raw_df)
    
    # Vérifications des résultats
    assert isinstance(processed_df, pd.DataFrame)
    assert not processed_df.empty
    assert "is_popular" in processed_df.columns
    assert "protected_is_english" in processed_df.columns
    assert "protected_is_major_studio" in processed_df.columns
    assert processed_df["protected_is_english"].iloc[0] == 1
    assert processed_df["protected_is_major_studio"].iloc[0] == 1