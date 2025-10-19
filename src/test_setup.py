import pytest
from data_preparation import load_data

def test_load_data():
    data = load_data('data/RAVDESS')  # Changed from '../data/RAVDESS'
    assert len(data) > 0, "Dataset should not be empty ok understtood"