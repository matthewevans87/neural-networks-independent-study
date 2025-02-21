import unittest
import torch
import torchvision
from src.data.data_loader import get_data_loader

class TestDataLoader(unittest.TestCase):
    def test_get_data_loader_train(self):
        # Test training data loader
        train_loader = get_data_loader(train=True)
        
        # Check if it's a DataLoader instance
        self.assertIsInstance(train_loader, torch.utils.data.DataLoader)
        
        # Check batch size
        self.assertEqual(train_loader.batch_size, 64)
        
        # # Check if shuffle is enabled
        # self.assertTrue(train_loader.shuffle)
        
        # Check dataset properties
        dataset = train_loader.dataset
        self.assertIsInstance(dataset, torchvision.datasets.MNIST)
        self.assertTrue(dataset.train)

    def test_get_data_loader_test(self):
        # Test test data loader
        test_loader = get_data_loader(train=False)
        
        # Check if it's a DataLoader instance
        self.assertIsInstance(test_loader, torch.utils.data.DataLoader)
        
        # Check batch size
        self.assertEqual(test_loader.batch_size, 64)
        
        # # Check if shuffle is enabled
        # self.assertTrue(test_loader.shuffle)
        
        # Check dataset properties
        dataset = test_loader.dataset
        self.assertIsInstance(dataset, torchvision.datasets.MNIST)
        self.assertFalse(dataset.train)

if __name__ == '__main__':
    unittest.main()
