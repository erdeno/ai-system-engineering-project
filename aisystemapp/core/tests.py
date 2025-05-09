from django.test import TestCase, Client
from django.urls import reverse

class AppTestCase(TestCase):
    def setUp(self):
        self.client = Client()

    def test_homepage_loads(self):
        """Ensure homepage loads correctly"""
        response = self.client.get(reverse('index'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Enter your review")

    def test_prediction_post(self):
        """Test POST request with valid text input"""
        test_input = "This product is amazing! I love it."
        response = self.client.post(reverse('index'), {'user_input': test_input})
        self.assertEqual(response.status_code, 200)
        self.assertIn('Positive', response.content.decode())

    def test_empty_input(self):
        """Test that empty input is handled gracefully"""
        response = self.client.post(reverse('index'), {'user_input': ''})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Enter your review")


class AppIntegrationTestCase(TestCase):
    def setUp(self):
        self.client = Client()

    def test_positive_review(self):
        response = self.client.post(reverse('index'), {
            'user_input': 'I absolutely loved this product! It exceeded my expectations in every way.'
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn('Positive', response.content.decode())

    def test_negative_review(self):
        response = self.client.post(reverse('index'), {
            'user_input': 'This is the worst item I have ever bought. Total waste of money.'
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn('Negative', response.content.decode())

    def test_neutral_mixed_review(self):
        response = self.client.post(reverse('index'), {
            'user_input': 'The item is okay. Some features are great, but others are disappointing.'
        })
        self.assertEqual(response.status_code, 200)
        self.assertTrue(
            'Positive' in response.content.decode() or 'Negative' in response.content.decode()
        )
