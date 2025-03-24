def age_cat(years):
    """
    Categorizes a given age into predefined age ranges.

    Args:
        years (int or float): The age to categorize.

    Returns:
        str: The corresponding age category based on the input age.
    """
    bins = [(0, 20, '0-20'), (21, 30, '20-30'), (31, 40, '30-40'),
            (41, 50, '40-50'), (51, 60, '50-60'), (61, 70, '60-70'), (71, float('inf'), '70+')]

    for lower, upper, category in bins:
        if lower <= years <= upper:
            return category
