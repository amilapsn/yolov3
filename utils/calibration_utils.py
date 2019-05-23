
def calculate_area_from_origin(pt1, pt2):
    """
    This function calculates the area of the triangle joining origin and the two given points.
    By ranking the areas we can get the order in which lines appear
    :param pt1: point1. if passed as an nd array should be of shape 2 x n
    :param pt2: point2. if passed as an nd array should be of shape 2 x n
    :return: area
    """
    # print(pt1.shape)
    return 0.5 * abs(pt1[0] * pt2[1] - pt1[1] * pt2[0])

