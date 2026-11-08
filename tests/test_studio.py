import io
import unittest
from unittest.mock import patch
import torch
from torch import nn
from studio.engine import adapters
from app import app


class StudioTests(unittest.TestCase):
    def test_adapter_learning_and_reload(self):
        torch.manual_seed(7)
        model = nn.ModuleDict({'attention': nn.ModuleDict({'q_proj': nn.Linear(4, 4)})})
        base = model['attention']['q_proj'].weight.detach().clone()
        params = adapters(model)
        optimizer = torch.optim.AdamW(params.values(), lr=.01)
        x = torch.randn(3, 4)
        before = model['attention']['q_proj'](x).detach().clone()
        for _ in range(2):
            optimizer.zero_grad()
            model['attention']['q_proj'](x).square().mean().backward()
            optimizer.step()
        after = model['attention']['q_proj'](x).detach().clone()
        self.assertFalse(torch.equal(before, after))
        self.assertTrue(torch.equal(base, model['attention']['q_proj'].base.weight))
        buffer = io.BytesIO()
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, buffer)
        buffer.seek(0)
        saved = torch.load(buffer, weights_only=True)
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        self.assertTrue(torch.equal(after, model['attention']['q_proj'](x)))

    @patch('app.busy', return_value=False)
    def test_local_api_validation(self, _busy):
        client = app.test_client()
        with client.get('/') as response:
            self.assertEqual(response.status_code, 200)
        self.assertEqual(client.get('/api/status').status_code, 200)
        self.assertEqual(client.post('/api/jobs', json={'action': 'train'}).status_code, 403)
        headers = {'X-Studio': 'local'}
        self.assertEqual(client.post('/api/jobs', json={'action': 'train', 'epochs': -1}, headers=headers).status_code, 400)
        self.assertEqual(client.post('/api/jobs', json={'action': 'generate', 'text': ''}, headers=headers).status_code, 400)
        self.assertEqual(client.get('/', headers={'Host': 'evil.example'}).status_code, 403)


if __name__ == '__main__':
    unittest.main()
