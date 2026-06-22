"""
多模态图片描述修复的单元测试
测试 _generate_image_description 的验证和重试逻辑

使用方法: conda run -n kbqa python test_multimodal_fix_v2.py
"""
import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# 确保项目根目录在 Python 路径中
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 显式导入模块，以使用 patch.object 方式 mock
import rag.retriever as retriever_mod
from rag.retriever import MultimodalRetriever
import httpx


class TestImageDescriptionValidation(unittest.TestCase):
    """测试图片描述生成的验证和重试逻辑"""

    def _create_retriever(self):
        """创建一个模拟的 MultimodalRetriever（绕过 __init__）"""
        r = MultimodalRetriever.__new__(MultimodalRetriever)
        r.embedder = MagicMock()
        r._client = MagicMock()
        r.collection_name = "test_mm"
        r._bm25_docs = []
        return r

    def _make_llm_response(self, text):
        """创建一个模拟的 LLM API 响应"""
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {"choices": [{"message": {"content": text}}]}
        return resp

    def _run_gen(self, retriever, max_retries=2):
        """调用 _generate_image_description"""
        return retriever._generate_image_description("base64data", "test.pdf", 1, max_retries=max_retries)

    # ---- 测试用例 ----

    def test_success_first_attempt(self):
        """第一次就成功生成有效描述"""
        with patch.object(retriever_mod, 'settings') as mock_settings, \
             patch.object(httpx, 'Client') as mock_client_class:
            mock_settings.llm_api_key = "test-key"
            mock_settings.mm_llm_api_url = "http://test.api/v1/chat/completions"
            mock_settings.mm_llm_model = "test-model"

            mock_client = MagicMock()
            mock_client.post.return_value = self._make_llm_response("这是一张展示山水风景的图片")
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client_class.return_value = mock_client

            r = self._create_retriever()
            result = self._run_gen(r)

            self.assertIn("这是一张展示山水风景的图片", result)
            self.assertIn("来源: test.pdf", result)
            self.assertIn("第1页", result)
            self.assertEqual(mock_client.post.call_count, 1)

    def test_retry_on_failure_response(self):
        """LLM 返回"无法看到图片"时自动重试"""
        with patch.object(retriever_mod, 'settings') as mock_settings, \
             patch.object(httpx, 'Client') as mock_client_class:
            mock_settings.llm_api_key = "test-key"
            mock_settings.mm_llm_api_url = "http://test.api/v1/chat/completions"
            mock_settings.mm_llm_model = "test-model"

            mock_client = MagicMock()
            mock_client.post.side_effect = [
                self._make_llm_response("很抱歉，我无法看到您提到的图片"),
                self._make_llm_response("蓝色的柱状统计图"),
            ]
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client_class.return_value = mock_client

            r = self._create_retriever()
            result = self._run_gen(r)

            self.assertIn("蓝色的柱状统计图", result)
            self.assertEqual(mock_client.post.call_count, 2)

    def test_fallback_after_all_retries_fail(self):
        """所有重试都失败后降级为默认标签"""
        with patch.object(retriever_mod, 'settings') as mock_settings, \
             patch.object(httpx, 'Client') as mock_client_class:
            mock_settings.llm_api_key = "test-key"
            mock_settings.mm_llm_api_url = "http://test.api/v1/chat/completions"
            mock_settings.mm_llm_model = "test-model"

            mock_client = MagicMock()
            mock_client.post.return_value = self._make_llm_response("抱歉，无法识别图片内容")
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client_class.return_value = mock_client

            r = self._create_retriever()
            result = self._run_gen(r)

            self.assertEqual(result, "[图片] 来源: test.pdf, 第1页")
            self.assertEqual(mock_client.post.call_count, 3)

    def test_retry_on_network_error(self):
        """网络异常时自动重试"""
        with patch.object(retriever_mod, 'settings') as mock_settings, \
             patch.object(httpx, 'Client') as mock_client_class:
            mock_settings.llm_api_key = "test-key"
            mock_settings.mm_llm_api_url = "http://test.api/v1/chat/completions"
            mock_settings.mm_llm_model = "test-model"

            mock_client = MagicMock()
            mock_client.post.side_effect = [
                Exception("Connection timeout"),
                self._make_llm_response("红色的饼图"),
            ]
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client_class.return_value = mock_client

            r = self._create_retriever()
            result = self._run_gen(r)

            self.assertIn("红色的饼图", result)
            self.assertEqual(mock_client.post.call_count, 2)

    def test_various_failure_keywords(self):
        """各种失败关键词都能被识别并触发重试/降级"""
        with patch.object(retriever_mod, 'settings') as mock_settings, \
             patch.object(httpx, 'Client') as mock_client_class:
            mock_settings.llm_api_key = "test-key"
            mock_settings.mm_llm_api_url = "http://test.api/v1/chat/completions"
            mock_settings.mm_llm_model = "test-model"

            failure_texts = [
                "无法识别图片",
                "我没有看到图片",
                "请重新上传图片",
                "cannot see the image",
                "我无法识别图片中的内容",
                "我没有办法看到你上传的图片",
                "无法看到您提到的图片",
            ]

            for fail_text in failure_texts:
                mock_client = MagicMock()
                mock_client.post.return_value = self._make_llm_response(fail_text)
                mock_client.__enter__ = MagicMock(return_value=mock_client)
                mock_client.__exit__ = MagicMock(return_value=False)
                mock_client_class.return_value = mock_client

                r = self._create_retriever()
                result = self._run_gen(r, max_retries=0)
                self.assertEqual(result, "[图片] 来源: test.pdf, 第1页",
                                 f"失败关键词 '{fail_text}' 未被识别")

    def test_valid_short_descriptions_accepted(self):
        """有效的短描述被正确接受（不会被误判为失败）"""
        with patch.object(retriever_mod, 'settings') as mock_settings, \
             patch.object(httpx, 'Client') as mock_client_class:
            mock_settings.llm_api_key = "test-key"
            mock_settings.mm_llm_api_url = "http://test.api/v1/chat/completions"
            mock_settings.mm_llm_model = "test-model"

            valid_descriptions = [
                "青色的折线统计图",
                "山水风景照片",
                "蓝色柱状图展示了销售数据",
                "一张表格，包含三列数据",
                "K线图",
            ]

            for desc_text in valid_descriptions:
                mock_client = MagicMock()
                mock_client.post.return_value = self._make_llm_response(desc_text)
                mock_client.__enter__ = MagicMock(return_value=mock_client)
                mock_client.__exit__ = MagicMock(return_value=False)
                mock_client_class.return_value = mock_client

                r = self._create_retriever()
                result = self._run_gen(r)

                self.assertIn(desc_text, result, f"有效描述 '{desc_text}' 被错误拒绝")


if __name__ == "__main__":
    unittest.main(verbosity=2)
