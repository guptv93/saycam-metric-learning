import torch
import numpy as np


class NPairLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity=True):
        super(NPairLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.mask_samples_from_same_repr = self._get_correlated_mask()
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            return self._cosine_similarity
        else:
            return self._dot_similarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).to(device=self.device, dtype=torch.bool)
        return mask
    
    def _cosine_similarity(self, x, y):
        # Use broadcasting to calculate cosine similarity between all vectors
        # https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065
        criterion = torch.nn.CosineSimilarity(dim=-1)
        v = criterion(x.unsqueeze(1), y.unsqueeze(0))
        return v


    def _dot_similarity(self, x, y):
        v = torch.mm(x, y.T)
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)
        
        top_p, top_class = logits.topk(1, dim=1)
        #print(top_class)
        equals = top_class == labels.view(*top_class.shape)
        accuracy = torch.mean(equals.type(torch.FloatTensor))

        return loss / (2 * self.batch_size), accuracy        