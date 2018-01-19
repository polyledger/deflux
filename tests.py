import deflux
import unittest


class DefluxTestCase(unittest.TestCase):

    def setUp(self):
        deflux.app.testing = True
        self.app = deflux.app.test_client()


if __name__ == '__main__':
    unittest.main()
