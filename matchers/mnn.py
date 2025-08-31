from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch as th
from torch import Tensor

# from libutils.utils_descriptors import get_margin_and_ratio_from_scores_and_mnn_matrix

def get_margin_and_ratio_from_scores_and_mnn_matrix(
        mnn_matrix: Tensor,
        best_scores0: Tensor,
        second_best_scores0: Tensor,
        second_best_scores1: Tensor
) -> tuple[Tensor, Tensor]:
    """
    Args:
        mnn_matrix:
            n0,n1 bool
        best_scores0:
            n0
        second_best_scores0:
            n0
        second_best_scores1:
            n1
    Returns:
        margin:
            n0
        ratio:
            n0
    """
    assert mnn_matrix.ndim == 2
    assert best_scores0.ndim == second_best_scores0.ndim == second_best_scores1.ndim == 1
    assert mnn_matrix.shape[0] == best_scores0.shape[0] == second_best_scores0.shape[0]
    assert mnn_matrix.shape[1] == second_best_scores1.shape[0]

    rows_matches_idx, column_matches_idx = th.where(mnn_matrix)  # (n_matches), (n_matches)
    best_scores0_matches = best_scores0[rows_matches_idx]  # n_matches_proposed
    # ? by definition of mnn, the best_scores0_matches are exactly the same as best_scores1_matches
    # best_scores1_matches = best_scores1[batch_matches, column_matches]  # n_matches_proposed
    second_best_scores0_matches = second_best_scores0[rows_matches_idx]  # n_matches_proposed
    second_best_scores1_matches = second_best_scores1[column_matches_idx]  # n_matches_proposed
    margin = best_scores0_matches - th.max(second_best_scores0_matches, second_best_scores1_matches)  # n_matches_proposed
    ratio = th.max(second_best_scores0_matches, second_best_scores1_matches) / best_scores0_matches  # n_matches_proposed
    return margin, ratio


@dataclass
class MatchingMatrixExtra:
    """ utility class to store the matching matrix and extra information about it """
    # ? the proposed matching matrix
    proposed: Tensor  # (B),n0,n1 bool
    # ? the matching matrix with the correct matches
    correct: Tensor | None = None  # (B),n0,n1 bool
    # ? the matching matrix with the wrong matches
    wrong: Tensor | None = None  # (B),n0,n1 bool
    # ? the matching matrix with the mismatched matches (there exist a correct matches for that point, but it's wrongly matched)
    mismatched: Tensor | None = None  # (B),n0,n1 bool
    # ? a match is found between two points that have no existing match in the GT_matching_matrix
    inexistent: Tensor | None = None  # (B),n0,n1 bool
    # ? matching_matrix_unsure is true when one of the proposed matches does not correspond either to a match or to an unmatch
    unsure: Tensor | None = None  # (B),n0,n1 bool
    score: Tensor | None = None  # (B),n0,n1 float

    def shape(self):
        return self.proposed.shape

    def __repr__(self):
        return f'MatchingMatrixExtra [{tuple(self.shape())}]  device: {self.proposed.device}'

    def __getitem__(self, b: int = 0):
        assert len(self.shape()) >= 3, 'MatchingMatrix must have at least 3 dimensions to be sliced'
        return MatchingMatrixExtra(
            proposed=self.proposed[b],
            correct=self.correct[b] if self.correct is not None else None,
            wrong=self.wrong[b] if self.wrong is not None else None,
            mismatched=self.mismatched[b] if self.mismatched is not None else None,
            inexistent=self.inexistent[b] if self.inexistent is not None else None,
            unsure=self.unsure[b] if self.unsure is not None else None,
            score=self.score[b] if self.score is not None else None
        )

    def to(self, device: str):
        self.proposed = self.proposed.to(device)
        self.correct = self.correct.to(device) if self.correct is not None else None
        self.wrong = self.wrong.to(device) if self.wrong is not None else None
        self.mismatched = self.mismatched.to(device) if self.mismatched is not None else None
        self.inexistent = self.inexistent.to(device) if self.inexistent is not None else None
        self.unsure = self.unsure.to(device) if self.unsure is not None else None
        self.score = self.score.to(device) if self.score is not None else None
        return self

    def cpu(self):
        return self.to('cpu')


@dataclass
class Matches:
    matches: Tensor  # n_matches,2
    score_matrix: Tensor  # n0,n1
    score_matrix_with_bins: Tensor | None = None  # n0+1,n1+1
    matching_matrix_extra: MatchingMatrixExtra | None = None
    matching_matrix_GT_with_bins: Tensor | None = None  # n0+1,n1+1

    @property
    def matching_matrix(self) -> Tensor:
        output = th.zeros_like(self.score_matrix, dtype=th.bool)
        output[self.matches[:, 0], self.matches[:, 1]] = True
        return output

    def _compute_matching_matrix_extra(self, matching_matrix_GT_with_bins: Tensor):
        self.matching_matrix_GT_with_bins = matching_matrix_GT_with_bins
        self.matching_matrix_extra = compute_correct_wrong_mismatched_inexistent_unsure_matches(self.matching_matrix[None], matching_matrix_GT_with_bins[None])[0]

    def compute_scores_stats(self, matching_matrix_GT_with_bins: Tensor) -> dict[str, float]:
        """ compute different matching statistics that can be useful to investigate the matching performance
        Args:
            matching_matrix_GT_with_bins:
                n0+1,n1+1
        """
        assert matching_matrix_GT_with_bins.ndim == 2, f'expected 2D tensor, got {matching_matrix_GT_with_bins.ndim}D'
        assert matching_matrix_GT_with_bins.shape[0] == self.score_matrix.shape[0] + 1, f'expected {self.score_matrix.shape[0] + 1} rows, got {matching_matrix_GT_with_bins.shape[0]}'
        assert matching_matrix_GT_with_bins.shape[1] == self.score_matrix.shape[1] + 1, f'expected {self.score_matrix.shape[1] + 1} cols, got {matching_matrix_GT_with_bins.shape[1]}'

        self._compute_matching_matrix_extra(matching_matrix_GT_with_bins)
        stats = {}

        n_matches_GT = matching_matrix_GT_with_bins[:-1, :-1].sum().item()
        n_matches_proposed = self.matching_matrix_extra.proposed.sum().item()
        n_matches_correct = self.matching_matrix_extra.correct.sum().item()
        stats['n_matches_GT'] = n_matches_GT
        stats['n_matches_proposed'] = n_matches_proposed
        stats['n_matches_correct'] = n_matches_correct
        stats['n_matches_wrong'] = self.matching_matrix_extra.wrong.sum().item()
        stats['n_matches_mismatched'] = self.matching_matrix_extra.mismatched.sum().item()
        stats['n_matches_inexistent'] = self.matching_matrix_extra.inexistent.sum().item()
        stats['n_matches_unsure'] = self.matching_matrix_extra.unsure.sum().item()

        stats['mean_GT_score'] = self.score_matrix[matching_matrix_GT_with_bins[:-1, :-1]].mean().item()
        stats['mean_proposed_score'] = self.score_matrix[self.matching_matrix_extra.proposed].mean().item()
        stats['mean_correct_score'] = self.score_matrix[self.matching_matrix_extra.correct].mean().item()
        stats['mean_wrong_score'] = self.score_matrix[self.matching_matrix_extra.wrong].mean().item()
        stats['mean_mismatched_score'] = self.score_matrix[self.matching_matrix_extra.mismatched].mean().item()
        stats['mean_inexistent_score'] = self.score_matrix[self.matching_matrix_extra.inexistent].mean().item()
        stats['mean_unsure_score'] = self.score_matrix[self.matching_matrix_extra.unsure].mean().item()
        stats['mean_matching_matrix_score'] = self.score_matrix.mean().item()

        stats['matches_precision'] = n_matches_correct / n_matches_proposed if n_matches_proposed > 0 else 0.0
        stats['matches_recall'] = n_matches_correct / n_matches_GT if n_matches_GT > 0 else 0.0

        # > compute the margins and ratios
        score_matrix_with_inf = self.score_matrix.clone()
        score_matrix_with_inf[score_matrix_with_inf.isnan()] = float('-inf')
        best_two_scores0 = th.topk(score_matrix_with_inf, 2, dim=-1)[0]  # (n0,2)
        best_two_scores1 = th.topk(score_matrix_with_inf, 2, dim=-2)[0].T  # (n1,2)
        best_scores0, second_best_scores0 = best_two_scores0[:, 0], best_two_scores0[:, 1]  # (n0), (n0)
        # ? best scores1 is not needed as all the matches are mutual nearest neighbors anyway,
        # ? so in all the following functions the sampled best_scores0 is the same as the sampled best_scores1 by definition
        _, second_best_scores1 = best_two_scores1[:, 0], best_two_scores1[:, 1]  # (n1), (n1)

        # ? margin for all the proposed matches
        margin_proposed, ratio_proposed = get_margin_and_ratio_from_scores_and_mnn_matrix(self.matching_matrix_extra.proposed, best_scores0, second_best_scores0, second_best_scores1)
        # ? correct matches margin
        margin_correct, ratio_correct = get_margin_and_ratio_from_scores_and_mnn_matrix(self.matching_matrix_extra.correct, best_scores0, second_best_scores0, second_best_scores1)
        # ? wrong matches margin
        margin_wrong, ratio_wrong = get_margin_and_ratio_from_scores_and_mnn_matrix(self.matching_matrix_extra.wrong, best_scores0, second_best_scores0, second_best_scores1)
        # ? mismatched matches margin
        margin_mismatched, ratio_mismatched = get_margin_and_ratio_from_scores_and_mnn_matrix(self.matching_matrix_extra.mismatched, best_scores0, second_best_scores0, second_best_scores1)
        # ? inexistent matches margin
        margin_inexistent, ratio_inexistent = get_margin_and_ratio_from_scores_and_mnn_matrix(self.matching_matrix_extra.inexistent, best_scores0, second_best_scores0, second_best_scores1)

        stats['mean_margin_proposed'] = margin_proposed.mean().item()
        stats['mean_margin_correct'] = margin_correct.mean().item()
        stats['mean_margin_wrong'] = margin_wrong.mean().item()
        stats['mean_margin_mismatched'] = margin_mismatched.mean().item()
        stats['mean_margin_inexistent'] = margin_inexistent.mean().item()

        stats['mean_ratio_proposed'] = ratio_proposed.mean().item()
        stats['mean_ratio_correct'] = ratio_correct.mean().item()
        stats['mean_ratio_wrong'] = ratio_wrong.mean().item()
        stats['mean_ratio_mismatched'] = ratio_mismatched.mean().item()
        stats['mean_ratio_inexistent'] = ratio_inexistent.mean().item()

        # > compute the n masked
        # ? find out how many possible mismatched have been shielded by a correct match
        # ? we do this counting how many column have the max score that correspond to a column where there is a correct match
        matches_correct_idx = self.matching_matrix_extra.correct.nonzero()  # n_matches_correct,2
        # ? we first create a mask with a one in the position where the score is the max for that row
        row_max_mask = \
            (score_matrix_with_inf == score_matrix_with_inf.max(dim=-1, keepdim=True)[0]) * \
            score_matrix_with_inf.isfinite()  # n0,n1
        # ? we then index only the columns where there was a correct match
        masked_columns = row_max_mask[:, matches_correct_idx[:, -1]].T  # n_masked_columns,n0
        # ? and sum over those columns (subtracting always one as we do not want to count the correct match)
        n_masked_by_columns = masked_columns.sum() - masked_columns.shape[0]
        # ? do the same by columns
        column_max_mask = \
            (score_matrix_with_inf == score_matrix_with_inf.max(dim=-2, keepdim=True)[0]) * \
            score_matrix_with_inf.isfinite()  # (n0,n1)
        masked_rows = column_max_mask[matches_correct_idx[:, -2], :]  # n_masked_rows, n1
        n_masked_by_rows = masked_rows.sum() - masked_rows.shape[0]
        n_masked = n_masked_by_columns + n_masked_by_rows
        stats['n_masked'] = n_masked.item()

        return stats

    def to(self, device: str) -> Matches:
        self.matches = self.matches.to(device)
        self.score_matrix = self.score_matrix.to(device)
        self.score_matrix_with_bins = self.score_matrix_with_bins.to(device) if self.score_matrix_with_bins is not None else None
        return self

    def cpu(self) -> Matches:
        return self.to('cpu')

    def __repr__(self):
        return f'Matches [{tuple(self.matches.shape)}]  device: {self.matches.device}'

    @property
    def shape(self) -> tuple[int, ...]:
        return self.score_matrix.shape


class Matcher(ABC):
    def __int__(self):
        super().__init__()
        self.name = 'Matcher'

    @abstractmethod
    def match(self, des0: list[Tensor], des1: list[Tensor]) -> list[Matches]:
        raise NotImplementedError

    @abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError


class MNN(Matcher):
    def __init__(self, min_score: float, ratio_test: float = 1.0, device: str = 'cpu'):
        self.min_score = min_score
        self.ratio_test = ratio_test
        self.device = device

        self.name = 'MNN'
        if self.min_score != -1.0:
            self.name = f'{self.name}{self.min_score}'
        if self.ratio_test != 1.0:
            self.name = f'{self.name}-ratiotest{self.ratio_test}'

    def match(self, des0: list[Tensor], des1: list[Tensor]) -> list[Matches]:
        matches_list, score_matrix_list = match_descriptors_mnn_scores_ratio_test(
            des0, des1, self.min_score, self.ratio_test)
        output = [Matches(matches, score_matrix) for matches, score_matrix in zip(matches_list, score_matrix_list)]
        return output

    def __repr__(self) -> str:
        return self.name


def mutual_nearest_neighbors_from_score_matrix(score_mat: Tensor, min_score: float = -1.0, ratio_test: float = 1.0) -> Tensor:
    """ return a boolean matrix with a True where the position was a maximum in both row and columns (and grater than the min score)
    Args:
        score_mat: score_matrix matrix
            Bxn0xn1
        min_score: minimum score to consider a match
        ratio_test: ratio test to apply to the score matrix
    Returns:
        mnn: mutual nearest neighbors matrix
            Bxn0xn1 th.bool
    """
    assert score_mat.ndim == 3
    B, n0, n1 = score_mat.shape
    if n0 == 0 or n1 == 0:
        return score_mat.new_zeros((B, n0, n1), dtype=th.bool)

    device = score_mat.device
    score_mat = score_mat.clone()
    score_mat[score_mat.isnan()] = float('-inf')

    # ? get the closest ones for each row and column
    # ? each row is the score between the descriptor from img0 and all the others in img1
    # ? each column is the score between the descriptor from img1 and all the others in img0
    nn0_value, nn0_idx = score_mat.max(2)  # (B,n0) (B,n0) with values [0, n1[
    nn1_value, nn1_idx = score_mat.max(1)  # (B,n1) (B,n1) with values [0, n0[

    nn0_idx[nn0_value == float('-inf')] = n1  # (B,n0) with values [0, n1]
    nn1_idx[nn1_value == float('-inf')] = n0  # (B,n1) with values [0, n0]

    nn0_matrix = th.zeros((B, n0+1, n1+1), dtype=th.bool, device=device)  # Bxn0xn1
    nn0_matrix[:, :-1, :].scatter_(2, nn0_idx[:, :, None], True)

    nn1_matrix = th.zeros((B, n0+1, n1+1), dtype=th.bool, device=device)  # Bxn0xn1
    nn1_matrix[:, :, :-1].scatter_(1, nn1_idx[:, None, :], True)

    # ? compose the two matrices
    mnn_matrix = nn0_matrix * nn1_matrix

    # ? drop the bins
    mnn_matrix = mnn_matrix[:, :-1, :-1]

    # ? remove the ones with score less than the min score
    mnn_matrix = mnn_matrix * (score_mat > min_score)

    if ratio_test < 1.0:
        best_scores0, idxs0 = score_mat.topk(2, dim=-1, largest=True, sorted=True)  # (B,n0,2) (B,n0,2)
        best_scores1, idxs1 = score_mat.topk(2, dim=-2, largest=True, sorted=True)  # (B,2,n1) (B,2,n1)
        valid_mask0 = best_scores0[:, :, 0] * ratio_test > best_scores0[:, :, 1]  # B,n0
        valid_mask1 = best_scores1[:, 0, :] * ratio_test > best_scores1[:, 1, :]  # n1
        ratio_test_mat = valid_mask0[:, :, None] * valid_mask1[:, None, :]
        mnn_matrix *= ratio_test_mat

    return mnn_matrix


def mutual_nearest_neighbors_from_dist_matrix(dist: Tensor) -> Tensor:
    """ return a boolean matrix with a True where the position was a minimum in both row and columns
    Args:
        dist: distance matrix
            Bxn0xn1
    Returns:
        mnn: mutual nearest neighbors matrix
            Bxn0xn1 th.bool
    """
    B, n0, n1 = dist.shape
    if n0 == 0 or n1 == 0:
        return dist.new_zeros((B, n0, n1), dtype=th.bool)

    device = dist.device

    # ? get the closest ones for each row and column
    nn0 = th.argmin(dist, dim=2)  # Bxn0 with values [0, n1[
    nn1 = th.argmin(dist, dim=1)  # Bxn1 with values [0, n0[

    # ? build the closest one matrix for each kpts0 (every row is dist from a kpts0_i and all the others kpts1)
    B0_idxs = th.arange(B).repeat_interleave(n0).to(device)  # B*n0
    nn0_matrix = th.zeros_like(dist, dtype=th.bool)  # Bxn0xn1
    nn0_matrix[B0_idxs, th.arange(n0).repeat(B, 1).reshape(-1), nn0.reshape(-1)] = True  # Bxn0xn1

    # ? build the closest one matrix for each kpts1 (every row is dist from a kpts1_i and all the others kpts0)
    B1_idxs = th.arange(B).repeat_interleave(n1)
    nn1_matrix = th.zeros_like(dist, dtype=th.bool)  # Bxn0xn1
    nn1_matrix[B1_idxs, nn1.reshape(-1), th.arange(n1).repeat(B, 1).reshape(-1)] = True

    # ? by multiplying the two matrices only the mutual-nearest-neighbours are selected
    mnn_matrix = nn0_matrix * nn1_matrix

    return mnn_matrix


def match_descriptors_mnn_scores_ratio_test(
        des0: list[Tensor],
        des1: list[Tensor],
        min_score: float = -1.0,
        ratio_test: float = 1.0
) -> tuple[list[Tensor], list[Tensor]]:
    """ match keypoints looking for mutual nearest neighbor in the descriptors space using the inner product
    Args:
        des0: list of descriptor tensors extracted from img0
            list[B] of Tensor[n_extracted0, des_dim]
        des1: list of descriptor tensors extracted from img1
            list[B] of Tensor[n_extracted1, des_dim]
        min_score: the minimum score of two mnn to be considered a valid match
        ratio_test: if > 0, we require the score of the second-best match to be at least ratio_test times smaller than the
    Returns:
        matches_list: list of matches given with double index notation
            list[B] of Tensor[n_matches, 2]     with order (idx0, idx1)
        score_matrix_list: list of score matrices
            list[B] of Tensor[n_extracted0, n_extracted1]
    """
    B = len(des0)
    device = des0[0].device

    matches_list = []
    score_matrix_list = []
    for b in range(B):
        # ? match keypoints
        if des0[b].shape[0] == 0 or des1[b].shape[0] == 0:
            matches = th.zeros(0, 2, device=device, dtype=th.long)
            score_matrix = th.zeros(des0[b].shape[0], des1[b].shape[0], device=device)
        else:
            score_matrix = des0[b] @ des1[b].permute(1, 0)  # n0 x n1
            # ? set the nan in the score_matrix to -1
            if score_matrix.isnan().any():
                print('WARNING: score matrix have nan values, setting those to -1')
                score_matrix[score_matrix.isnan()] = -1
            matches_mat = mutual_nearest_neighbors_from_score_matrix(
                score_matrix[None], min_score=min_score, ratio_test=ratio_test)[0]  # n0 x n1

            matches = th.nonzero(matches_mat)
        matches_list.append(matches)
        score_matrix_list.append(score_matrix)
    return matches_list, score_matrix_list


def compute_correct_wrong_mismatched_inexistent_unsure_matches(matching_matrix: Tensor, GT_matching_matrix_with_bins: Tensor)\
        -> MatchingMatrixExtra:
    """
    Args:
        matching_matrix: the matching matrix obtained from descriptors
            B,n0,n1
        GT_matching_matrix_with_bins: the GT matching matrix with one additional bin row and column with the unmatched keypoints
            B,n0+1,n1+1

    Returns:
        MatchingMatrixExtra
    """
    assert matching_matrix.shape[0] == GT_matching_matrix_with_bins.shape[0], f'{matching_matrix.shape[0]} != {GT_matching_matrix_with_bins.shape[0]}'
    assert matching_matrix.shape[1] == GT_matching_matrix_with_bins.shape[1] - 1, f'{matching_matrix.shape[1]} != {GT_matching_matrix_with_bins.shape[1] - 1}'
    assert matching_matrix.shape[2] == GT_matching_matrix_with_bins.shape[2] - 1, f'{matching_matrix.shape[2]} != {GT_matching_matrix_with_bins.shape[2] - 1}'
    assert matching_matrix.ndim == 3, f'{matching_matrix.ndim} != 3'
    assert matching_matrix.dtype == th.bool and GT_matching_matrix_with_bins.dtype == th.bool, f'{matching_matrix.dtype} != {th.bool} or {GT_matching_matrix_with_bins.dtype} != {th.bool}'

    GT_matching_matrix = GT_matching_matrix_with_bins[:, :-1, :-1]
    B, H, W = GT_matching_matrix.shape

    matching_matrix_correct = matching_matrix * GT_matching_matrix

    # ? known_mask is true for each row and column where there is a one, either as match or in the bin
    known_mask_with_bins = GT_matching_matrix_with_bins.any(1, keepdim=True).repeat(1, H + 1, 1) + GT_matching_matrix_with_bins.any(2, keepdim=True).repeat(1, 1, W + 1)
    known_mask = known_mask_with_bins[:, :-1, :-1]

    # ? match_mask is true for each row and column where there is a GT match
    any_match_mask = GT_matching_matrix.any(1, keepdim=True).repeat(1, H, 1) + GT_matching_matrix.any(2, keepdim=True).repeat(1, 1, W)

    # ? matching_matrix_unsure is true when one of the proposed matches does not correspond either to a match or to an unmatch
    matching_matrix_unsure = matching_matrix * ~known_mask

    # ? matching_matrix_wrong is true when a proposed match is wrong (either a mismatch or inexistent)
    matching_matrix_wrong = ((matching_matrix ^ GT_matching_matrix) * matching_matrix) * known_mask

    # ? mismatch_mask is true when a point that actually had a possible correct match is mismatched
    matching_matrix_mismatched = matching_matrix_wrong * any_match_mask

    # ? inexistent_mask is true when two keypoints that had not GT match are matched
    matching_matrix_inexistent = matching_matrix_wrong * ~any_match_mask

    output = MatchingMatrixExtra(
        matching_matrix,
        matching_matrix_correct,
        matching_matrix_wrong,
        matching_matrix_mismatched,
        matching_matrix_inexistent,
        matching_matrix_unsure
    )
    return output


# from DeDode

import torch
from PIL import Image
import torch.nn as nn
import torchvision.models as tvm
import torch.nn.functional as F
import numpy as np


def dual_softmax_matcher(desc_A: tuple['B','N','C'], desc_B: tuple['B','M','C'], inv_temperature = 1, normalize = False):
    if len(desc_A.shape) < 3:
        desc_A, desc_B = desc_A[None], desc_B[None]
    B, N, C = desc_A.shape
    if normalize:
        desc_A = desc_A/desc_A.norm(dim=-1,keepdim=True)
        desc_B = desc_B/desc_B.norm(dim=-1,keepdim=True)
        corr = torch.einsum("b n c, b m c -> b n m", desc_A, desc_B) * inv_temperature
    else:
        corr = torch.einsum("b n c, b m c -> b n m", desc_A, desc_B) * inv_temperature
    P = corr.softmax(dim = -2) * corr.softmax(dim= -1)
    return P

def to_pixel_coords(flow, h1, w1):
    flow = (
        torch.stack(
            (
                w1 * (flow[..., 0] + 1) / 2,
                h1 * (flow[..., 1] + 1) / 2,
            ),
            axis=-1,
        )
    )
    return flow

def to_normalized_coords(flow, h1, w1):
    flow = (
        torch.stack(
            (
                2 * (flow[..., 0]) / w1 - 1,
                2 * (flow[..., 1]) / h1 - 1,
            ),
            axis=-1,
        )
    )
    return flow

class DualSoftMaxMatcher(nn.Module):        
    @torch.inference_mode()
    def match(self, keypoints_A, descriptions_A, 
              keypoints_B, descriptions_B, P_A = None, P_B = None, 
              normalize = False, inv_temp = 1, threshold = 0.0):
        if isinstance(descriptions_A, list):
            matches = [self.match(k_A[None], d_A[None], k_B[None], d_B[None], normalize = normalize,
                               inv_temp = inv_temp, threshold = threshold) 
                    for k_A,d_A,k_B,d_B in
                    zip(keypoints_A, descriptions_A, keypoints_B, descriptions_B)]
            matches_A = torch.cat([m[0] for m in matches])
            matches_B = torch.cat([m[1] for m in matches])
            inds = torch.cat([m[2] + b for b, m in enumerate(matches)])
            return matches_A, matches_B, inds
        
        P = dual_softmax_matcher(descriptions_A, descriptions_B, 
                                 normalize = normalize, inv_temperature=inv_temp, # corr.softmax(dim = -2) * corr.softmax(dim= -1)
                                 )
        inds = torch.nonzero((P == P.max(dim=-1, keepdim = True).values) 
                        * (P == P.max(dim=-2, keepdim = True).values) * (P > threshold))
        batch_inds = inds[:,0]
        matches_A = keypoints_A[batch_inds, inds[:,1]]
        matches_B = keypoints_B[batch_inds, inds[:,2]]
        return matches_A, matches_B, batch_inds

    def to_pixel_coords(self, x_A, x_B, H_A, W_A, H_B, W_B):
        return to_pixel_coords(x_A, H_A, W_A), to_pixel_coords(x_B, H_B, W_B)
    
    def to_normalized_coords(self, x_A, x_B, H_A, W_A, H_B, W_B):
        return to_normalized_coords(x_A, H_A, W_A), to_normalized_coords(x_B, H_B, W_B)
    