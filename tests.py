import api

rectangles = itertools.islice(api.gen_rectangle(((-10, 1), (-8, 1), (-8, 0), (-10, 0)), 0.5), 7)
api.visualise_polygons(rectangles)