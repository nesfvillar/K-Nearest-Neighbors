import unittest

from knearest.knearest import KNearest


class TestKNearest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.knn = KNearest([
            ("Orange", (1, 8)),
            ("Orange", (2, 7)),
            ("Tomato", (8, 2))
        ])

    def test_get_data_points(self):
        self.assertEqual([("Orange", (1, 8)), ("Orange", (2, 7)), ("Tomato", (8, 2))], self.knn.get_data_points())

    def test_add_data_point(self):
        new_point = "Lemon", (7, 5)
        self.assertNotIn(new_point, self.knn.get_data_points())
        self.knn.add_data_point(new_point)
        self.assertIn(new_point, self.knn.get_data_points())

    def test_remove_data_point(self):
        point = "Tomato", (8, 2)
        self.assertIn(point, self.knn.get_data_points())
        self.knn.remove_data_point(point)
        self.assertNotIn(point, self.knn.get_data_points())

    def test_get_nearest_k(self):
        k = 3
        closest_k = self.knn.get_nearest_k((7, 3), k=k)
        self.assertEqual(3, len(closest_k))

    def test_classify(self):
        closest_kind, closest_count = self.knn.classify((8, 2), k=2)
        self.assertEqual("Tomato", closest_kind)
        self.assertEqual(1, closest_count)

        closest_kind, closest_count = self.knn.classify((8, 2), k=3)
        self.assertEqual("Orange", closest_kind)
        self.assertEqual(2, closest_count)


if __name__ == '__main__':
    unittest.main()
