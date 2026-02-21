"""
Group 6 - Feature Engineering
tests/test_functions.py — PyTest test cases for all 5 processing functions
Run with: pytest tests/test_functions.py -v
"""

import pytest
import pandas as pd
import os
import sys

# Add parent directory to path so imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from derive_computed_columns       import derive_computed_columns
from encode_categorical_features   import encode_categorical_features
from bin_numeric_ranges            import bin_numeric_ranges
from time_based_feature_extraction import time_based_feature_extraction
from flag_anomalies_column         import flag_anomalies_column

# ─── Shared fixtures ──────────────────────────────────────────────────────────
INPUT  = "input/data.csv"
OUTPUT = "output/"

@pytest.fixture(scope="session", autouse=True)
def ensure_output_dir():
    os.makedirs(OUTPUT, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# FUNCTION 1: derive_computed_columns
# ═══════════════════════════════════════════════════════════════════════════════

class TestDeriveComputedColumns:

    @pytest.fixture(scope="class")
    def df(self):
        return derive_computed_columns(INPUT, f"{OUTPUT}test_derived.csv")

    def test_output_file_created(self, df):
        """Output CSV file should exist after function runs."""
        assert os.path.exists(f"{OUTPUT}test_derived.csv")

    def test_salary_per_age_column_exists(self, df):
        """salary_per_age column should be created."""
        assert 'salary_per_age' in df.columns

    def test_annual_bonus_column_exists(self, df):
        """annual_bonus column should be created."""
        assert 'annual_bonus' in df.columns

    def test_is_senior_column_exists(self, df):
        """is_senior column should be created."""
        assert 'is_senior' in df.columns

    def test_salary_level_column_exists(self, df):
        """salary_level column should be created."""
        assert 'salary_level' in df.columns

    def test_score_rank_column_exists(self, df):
        """score_rank column should be created."""
        assert 'score_rank' in df.columns

    def test_is_senior_binary(self, df):
        """is_senior should only contain 0 or 1."""
        assert set(df['is_senior'].unique()).issubset({0, 1})

    def test_is_senior_correct_for_age_40_plus(self, df):
        """Anyone aged 40+ should have is_senior = 1."""
        senior_rows = df[df['age'] >= 40]
        assert (senior_rows['is_senior'] == 1).all()

    def test_is_senior_correct_for_under_40(self, df):
        """Anyone under 40 should have is_senior = 0."""
        junior_rows = df[df['age'] < 40]
        assert (junior_rows['is_senior'] == 0).all()

    def test_salary_per_age_positive(self, df):
        """salary_per_age should be positive for all rows."""
        assert (df['salary_per_age'] > 0).all()

    def test_annual_bonus_is_10_percent(self, df):
        """Annual bonus should be exactly 10% of salary."""
        original = pd.read_csv(INPUT)
        expected = (original['salary'] * 0.10).round(2)
        pd.testing.assert_series_equal(df['annual_bonus'].reset_index(drop=True),
                                       expected.reset_index(drop=True))

    def test_salary_level_valid_values(self, df):
        """salary_level should only contain: High, Mid, Low."""
        assert set(df['salary_level'].unique()).issubset({'High', 'Mid', 'Low'})

    def test_no_null_values_in_derived(self, df):
        """Derived columns should not have null values."""
        new_cols = ['salary_per_age', 'annual_bonus', 'is_senior', 'salary_level', 'score_rank']
        assert df[new_cols].isnull().sum().sum() == 0

    def test_row_count_preserved(self, df):
        """Row count should match the input file."""
        original = pd.read_csv(INPUT)
        assert len(df) == len(original)


# ═══════════════════════════════════════════════════════════════════════════════
# FUNCTION 2: encode_categorical_features
# ═══════════════════════════════════════════════════════════════════════════════

class TestEncodeCategoricalFeatures:

    @pytest.fixture(scope="class")
    def df(self):
        return encode_categorical_features(INPUT, f"{OUTPUT}test_encoded.csv")

    def test_output_file_created(self, df):
        """Output CSV file should exist after function runs."""
        assert os.path.exists(f"{OUTPUT}test_encoded.csv")

    def test_department_one_hot_columns_exist(self, df):
        """One-hot encoded department columns should be created."""
        assert any(col.startswith('dept_') for col in df.columns)

    def test_original_department_column_removed(self, df):
        """Original 'department' column should be removed after encoding."""
        assert 'department' not in df.columns

    def test_category_encoded_column_exists(self, df):
        """category_encoded column should be created."""
        assert 'category_encoded' in df.columns

    def test_category_encoded_valid_values(self, df):
        """category_encoded should only contain: 1, 2, 3."""
        assert set(df['category_encoded'].unique()).issubset({1, 2, 3})

    def test_category_encoded_no_nulls(self, df):
        """category_encoded should have no null values."""
        assert df['category_encoded'].isnull().sum() == 0

    def test_dept_columns_are_binary(self, df):
        """One-hot dept_ columns should only contain 0 or 1."""
        dept_cols = [col for col in df.columns if col.startswith('dept_')]
        for col in dept_cols:
            assert set(df[col].unique()).issubset({0, 1})

    def test_each_row_has_exactly_one_dept(self, df):
        """Each row should belong to exactly one department (sum of dept_ cols = 1)."""
        dept_cols = [col for col in df.columns if col.startswith('dept_')]
        assert (df[dept_cols].sum(axis=1) == 1).all()

    def test_row_count_preserved(self, df):
        """Row count should match the input file."""
        original = pd.read_csv(INPUT)
        assert len(df) == len(original)


# ═══════════════════════════════════════════════════════════════════════════════
# FUNCTION 3: bin_numeric_ranges
# ═══════════════════════════════════════════════════════════════════════════════

class TestBinNumericRanges:

    @pytest.fixture(scope="class")
    def df(self):
        return bin_numeric_ranges(INPUT, f"{OUTPUT}test_binned.csv")

    def test_output_file_created(self, df):
        """Output CSV file should exist after function runs."""
        assert os.path.exists(f"{OUTPUT}test_binned.csv")

    def test_age_group_column_exists(self, df):
        """age_group column should be created."""
        assert 'age_group' in df.columns

    def test_salary_range_column_exists(self, df):
        """salary_range column should be created."""
        assert 'salary_range' in df.columns

    def test_score_grade_column_exists(self, df):
        """score_grade column should be created."""
        assert 'score_grade' in df.columns

    def test_age_group_valid_labels(self, df):
        """age_group should only have valid label values."""
        valid = {'Young', 'Adult', 'Mid-Age', 'Senior'}
        actual = set(df['age_group'].dropna().unique())
        assert actual.issubset(valid)

    def test_salary_range_valid_labels(self, df):
        """salary_range should only have valid label values."""
        valid = {'Entry', 'Mid', 'Senior', 'Executive'}
        actual = set(df['salary_range'].dropna().unique())
        assert actual.issubset(valid)

    def test_score_grade_valid_labels(self, df):
        """score_grade should only have valid label values."""
        valid = {'Fail', 'Pass', 'Good', 'Excellent'}
        actual = set(df['score_grade'].dropna().unique())
        assert actual.issubset(valid)

    def test_no_extra_columns_removed(self, df):
        """Original columns should still be present."""
        original = pd.read_csv(INPUT)
        for col in original.columns:
            assert col in df.columns

    def test_row_count_preserved(self, df):
        """Row count should match the input file."""
        original = pd.read_csv(INPUT)
        assert len(df) == len(original)


# ═══════════════════════════════════════════════════════════════════════════════
# FUNCTION 4: time_based_feature_extraction
# ═══════════════════════════════════════════════════════════════════════════════

class TestTimeBasedFeatureExtraction:

    @pytest.fixture(scope="class")
    def df(self):
        return time_based_feature_extraction(INPUT, f"{OUTPUT}test_time.csv")

    def test_output_file_created(self, df):
        """Output CSV file should exist after function runs."""
        assert os.path.exists(f"{OUTPUT}test_time.csv")

    def test_join_year_column_exists(self, df):
        """join_year column should be created."""
        assert 'join_year' in df.columns

    def test_join_month_column_exists(self, df):
        """join_month column should be created."""
        assert 'join_month' in df.columns

    def test_join_quarter_column_exists(self, df):
        """join_quarter column should be created."""
        assert 'join_quarter' in df.columns

    def test_years_in_company_column_exists(self, df):
        """years_in_company column should be created."""
        assert 'years_in_company' in df.columns

    def test_is_recent_hire_column_exists(self, df):
        """is_recent_hire column should be created."""
        assert 'is_recent_hire' in df.columns

    def test_join_year_valid_range(self, df):
        """join_year should be between 2000 and current year."""
        import datetime
        current_year = datetime.datetime.today().year
        assert df['join_year'].between(2000, current_year).all()

    def test_join_month_valid_range(self, df):
        """join_month should be between 1 and 12."""
        assert df['join_month'].between(1, 12).all()

    def test_join_quarter_valid_range(self, df):
        """join_quarter should be between 1 and 4."""
        assert df['join_quarter'].between(1, 4).all()

    def test_years_in_company_positive(self, df):
        """years_in_company should be positive for all rows."""
        assert (df['years_in_company'] > 0).all()

    def test_is_recent_hire_binary(self, df):
        """is_recent_hire should only contain 0 or 1."""
        assert set(df['is_recent_hire'].unique()).issubset({0, 1})

    def test_row_count_preserved(self, df):
        """Row count should match the input file."""
        original = pd.read_csv(INPUT)
        assert len(df) == len(original)


# ═══════════════════════════════════════════════════════════════════════════════
# FUNCTION 5: flag_anomalies_column
# ═══════════════════════════════════════════════════════════════════════════════

class TestFlagAnomaliesColumn:

    @pytest.fixture(scope="class")
    def df(self):
        return flag_anomalies_column(INPUT, f"{OUTPUT}test_flagged.csv")

    def test_output_file_created(self, df):
        """Output CSV file should exist after function runs."""
        assert os.path.exists(f"{OUTPUT}test_flagged.csv")

    def test_salary_anomaly_column_exists(self, df):
        """salary_anomaly column should be created."""
        assert 'salary_anomaly' in df.columns

    def test_score_anomaly_column_exists(self, df):
        """score_anomaly column should be created."""
        assert 'score_anomaly' in df.columns

    def test_age_anomaly_column_exists(self, df):
        """age_anomaly column should be created."""
        assert 'age_anomaly' in df.columns

    def test_is_anomaly_column_exists(self, df):
        """is_anomaly column should be created."""
        assert 'is_anomaly' in df.columns

    def test_salary_anomaly_binary(self, df):
        """salary_anomaly should only contain 0 or 1."""
        assert set(df['salary_anomaly'].unique()).issubset({0, 1})

    def test_score_anomaly_binary(self, df):
        """score_anomaly should only contain 0 or 1."""
        assert set(df['score_anomaly'].unique()).issubset({0, 1})

    def test_is_anomaly_binary(self, df):
        """is_anomaly should only contain 0 or 1."""
        assert set(df['is_anomaly'].unique()).issubset({0, 1})

    def test_is_anomaly_is_union_of_flags(self, df):
        """is_anomaly should be 1 whenever any individual flag is 1."""
        expected = (
            (df['salary_anomaly'] == 1) |
            (df['score_anomaly']  == 1) |
            (df['age_anomaly']    == 1)
        ).astype(int)
        pd.testing.assert_series_equal(df['is_anomaly'], expected, check_names=False)

    def test_no_null_anomaly_flags(self, df):
        """All anomaly flag columns should have no null values."""
        flag_cols = ['salary_anomaly', 'score_anomaly', 'age_anomaly', 'is_anomaly']
        assert df[flag_cols].isnull().sum().sum() == 0

    def test_row_count_preserved(self, df):
        """Row count should match the input file."""
        original = pd.read_csv(INPUT)
        assert len(df) == len(original)
