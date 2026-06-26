import os
import unittest
from unittest.mock import MagicMock, patch
from PIL import Image

# Add parent directory to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_manager import GPUManager, ModelManager

class TestResolutionLogic(unittest.TestCase):
    def test_vram_resolution_classifications(self):
        # 1. Test 4GB override
        with patch.dict(os.environ, {"FORCE_4GB_OPTIMIZATION": "1", "FORCE_8GB_OPTIMIZATION": "0", "FORCE_12GB_OPTIMIZATION": "0"}):
            gm = GPUManager()
            self.assertTrue(gm.is_4gb_gpu)
            self.assertEqual(gm.max_dimension, 512)

        # 2. Test 8GB override
        with patch.dict(os.environ, {"FORCE_4GB_OPTIMIZATION": "0", "FORCE_8GB_OPTIMIZATION": "1", "FORCE_12GB_OPTIMIZATION": "0"}):
            gm = GPUManager()
            self.assertFalse(gm.is_4gb_gpu)
            self.assertEqual(gm.max_dimension, 768)

        # 3. Test 12GB override
        with patch.dict(os.environ, {"FORCE_4GB_OPTIMIZATION": "0", "FORCE_8GB_OPTIMIZATION": "0", "FORCE_12GB_OPTIMIZATION": "1"}):
            gm = GPUManager()
            self.assertFalse(gm.is_4gb_gpu)
            self.assertEqual(gm.max_dimension, 1024)

    def test_proportional_aspect_ratio_calculations(self):
        mm = ModelManager()
        
        # Test 1920x1080 under 4GB mode (longest side <= 512) -> should be 512x288
        w, h = mm.get_proportional_size((1920, 1080), 512)
        self.assertEqual((w, h), (512, 288))
        self.assertEqual(w % 8, 0)
        self.assertEqual(h % 8, 0)

        # Test 1920x1080 under 8GB mode (longest side <= 768) -> should be 768x432
        w, h = mm.get_proportional_size((1920, 1080), 768)
        self.assertEqual((w, h), (768, 432))
        self.assertEqual(w % 8, 0)
        self.assertEqual(h % 8, 0)

        # Test 1920x1080 under 12GB+ mode (longest side <= 1024) -> should be 1024x576
        w, h = mm.get_proportional_size((1920, 1080), 1024)
        self.assertEqual((w, h), (1024, 576))
        self.assertEqual(w % 8, 0)
        self.assertEqual(h % 8, 0)

        # Test vertical image: 1080x1920 under 4GB mode -> should be 288x512
        w, h = mm.get_proportional_size((1080, 1920), 512)
        self.assertEqual((w, h), (288, 512))

    @patch('model_manager.ModelManager.get_ghibli_pipe')
    def test_run_ghibli_inference_proportional_resizing(self, mock_get_pipe):
        mock_pipe = MagicMock()
        mock_output = MagicMock()
        mock_output.images = [Image.new("RGB", (100, 100))]
        mock_pipe.return_value = mock_output
        mock_get_pipe.return_value = mock_pipe

        mm = ModelManager()

        # 1920x1080 input image
        input_img = Image.new("RGB", (1920, 1080))

        # Test Case A: 4GB Mode (should resize to 512x288)
        with patch('model_manager.gpu_manager.is_4gb_gpu', True), \
             patch('model_manager.gpu_manager.max_dimension', 512):
            
            mm.run_ghibli_inference("test prompt", input_img)
            
            called_args = mock_pipe.call_args[1]
            called_image = called_args['image']
            self.assertEqual(called_image.size, (512, 288))

        # Test Case B: 8GB Mode (should resize to 768x432)
        with patch('model_manager.gpu_manager.is_4gb_gpu', False), \
             patch('model_manager.gpu_manager.max_dimension', 768):
            
            mm.run_ghibli_inference("test prompt", input_img)
            
            called_args = mock_pipe.call_args[1]
            called_image = called_args['image']
            self.assertEqual(called_image.size, (768, 432))

        # Test Case C: 12GB+ Mode (should resize to 1024x576)
        with patch('model_manager.gpu_manager.is_4gb_gpu', False), \
             patch('model_manager.gpu_manager.max_dimension', 1024):
            
            mm.run_ghibli_inference("test prompt", input_img)
            
            called_args = mock_pipe.call_args[1]
            called_image = called_args['image']
            self.assertEqual(called_image.size, (1024, 576))

if __name__ == '__main__':
    unittest.main()
