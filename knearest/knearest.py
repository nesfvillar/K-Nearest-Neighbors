from collections import Counter
import heapq
import math


class KNearest:
    def __init__(self, data_points: [tuple[any, tuple]]):
        self._data_points = data_points

    def add_data_point(self, data_point: [any, tuple]):
        self._data_points.append(data_point)

    def remove_data_point(self, data_point: [any, tuple]):
        self._data_points.remove(data_point)

    def get_data_points(self) -> [tuple[any, tuple]]:
        return self._data_points

    def get_nearest_k(self, location: [], formula=math.dist, k=3) -> [float, [tuple[any, tuple]]]:
        assert len(location) == len(self._data_points[0][1])

        h = []
        for point in self._data_points:
            distance = formula(location, point[1])
            heapq.heappush(h, (distance, point))
        return heapq.nsmallest(k, h)

    def classify(self, location: [], **kwargs):
        closest_kinds = map(lambda p: p[1][0], self.get_nearest_k(location, **kwargs))
        counter = Counter(closest_kinds)
        return counter.most_common()[0]

    def __repr__(self):
        return repr(self.get_data_points())
