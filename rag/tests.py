from django.test import TestCase
from django.urls import reverse

from .models import Document


class EmbeddingStatusApiTests(TestCase):
    def test_returns_recent_documents_with_progress_fields(self):
        doc = Document.objects.create(
            file='documents/sample.pdf',
            filename='sample.pdf',
            file_type='pdf',
            embed_status='processing',
            embed_progress=44,
            embed_step='Đang nhúng: 11/25 đoạn',
            embedding_model='nomic-embed-text',
            vector_db_key='nomic_v1_db',
        )

        response = self.client.get(reverse('embedding_status_api'))
        self.assertEqual(response.status_code, 200)

        payload = response.json()
        self.assertIn('documents', payload)
        self.assertTrue(payload['documents'])

        first_item = payload['documents'][0]
        self.assertEqual(first_item['id'], doc.id)
        self.assertEqual(first_item['embed_status'], 'processing')
        self.assertEqual(first_item['embed_progress'], 44)
        self.assertEqual(first_item['embed_step'], 'Đang nhúng: 11/25 đoạn')

    def test_can_filter_by_document_id(self):
        doc_keep = Document.objects.create(
            file='documents/a.pdf',
            filename='a.pdf',
            file_type='pdf',
            embed_status='pending',
            embedding_model='nomic-embed-text',
            vector_db_key='nomic_v1_db',
        )
        Document.objects.create(
            file='documents/b.pdf',
            filename='b.pdf',
            file_type='pdf',
            embed_status='completed',
            is_embedded=True,
            embedding_model='qwen3-embedding:0.6b',
            vector_db_key='qwen_db',
        )

        response = self.client.get(reverse('embedding_status_api'), {'document_id': doc_keep.id})
        self.assertEqual(response.status_code, 200)

        payload = response.json()
        self.assertEqual(len(payload['documents']), 1)
        self.assertEqual(payload['documents'][0]['id'], doc_keep.id)
