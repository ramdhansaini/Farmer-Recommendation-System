import unittest
from app import app


class FarmerAppTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_crop_form_posts_to_predict_endpoint(self):
        response = self.client.get('/crop_recommendation')
        self.assertEqual(response.status_code, 200)
        self.assertIn('action="/predict_crop"', response.get_data(as_text=True))

    def test_fertilizer_form_posts_to_predict_endpoint(self):
        response = self.client.get('/fertilizer_recommendation')
        self.assertEqual(response.status_code, 200)
        self.assertIn('action="/predict_fertilizer"', response.get_data(as_text=True))


if __name__ == '__main__':
    unittest.main()
