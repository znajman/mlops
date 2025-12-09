"""
Feature schema for the UCI Car Evaluation dataset.
Columns in the CSV:
buying, maint, doors, persons, lug_boot, safety, class
"""

# Car dataset has no continuous or discrete numeric features
CONTINOUS_COLUMNS = []
DISCRETE_COLUMNS = []

# All input features are nominal (categorical)
NOMINAL_COLUMNS = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']

# Target column name
TARGET_COLUMN = 'class'
