import triton
import torch
from torch.nn import functional as F


def prepare_nats_block_indices(
    n_nats_blocks: torch.LongTensor,
    nats_block_size: int,
    chunk_size: int,
) -> torch.LongTensor:
    """
    TODO this is only a Temporary solution, we could check if there is any better solution for that
    prepeare the chunk indices for nats. We notice that since the nats_block_size might be different for different nats
    heads, instead of returning an tensor of shape [n_blocks, 2] where the first value indicates the batch indices and
    the second value indicates the length indices, we will have  the first value indicates the batch * nats_heads indices
    Args:
        n_nats_blocks: torch.Tensor
            number of nats chunk for each operation, a tensor of shape [B, HNAtS, n_opts]
        nats_block_size: int
            nats chunk size
        chunk_size: int
            computational block size, BT
        op_idx: int
            index of the operation
    Returns:

    """
    indices = torch.cat(
        [torch.arange(n) for n in triton.cdiv(n_nats_blocks * nats_block_size, chunk_size).flatten()]
    )
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(n_nats_blocks)


def prepare_nats_chunk_offsets(
    n_nats_blocks: torch.LongTensor,
    nats_block_types: torch.Tensor,
    nats_block_size: int,
    chunk_size: int,
    op_idx: int,
) -> torch.LongTensor:
    """
    prepare the nats chunk offsets
    Args:
        n_nats_blocks: torch.Tensor,
        nats_block_types: torch.Tensor
        nats_block_size:
        chunk_size:
        op_idx:

    Returns:
        object:
        the offset for each heads stored in H

    """
    n_chunks = triton.cdiv(n_nats_blocks[..., op_idx] * nats_block_size, chunk_size).flatten()
    return F.pad(n_chunks, (1,0), mode='constant', value=0).cumsum(0)

    n_chunks = triton.cdiv(n_nats_blocks[..., op_idx] * nats_block_size, chunk_size).flatten()
    # if the last block is not the corresponding block, we also need to store those since they might be required
    n_chunks_h = n_chunks + torch.where(nats_block_types[:, -1, :, op_idx] == 0., 1, 0).flatten()
    n_chunks_info = F.pad(torch.stack([n_chunks, n_chunks_h], dim=1), (0, 0, 1,0),  mode='constant', value=0)
    return n_chunks_info.cumsum(0)


def compute_attn_n_iters_per_block(nats_chunk_indices: torch.Tensor,
                                   T: int,
                                   BT: int,
                                   BS: int,
                                   NAtS_Block_Size: int,
                                   attn_offset: int,
                                   n_data_in_current_chunk: int=0,
                                   sliding_window_size:int | None = None,
                                   ):
    """
    This function is used to compute how many full iterations we need to compute for each attention workers
    (i_t)
    Args:
        nats_chunk_indices(torch.Tensor):
            a torch tensor of shape [B, HNAtS, T_NAtS], indicating which columns are attention columns
        T (int):
            input context length
        BT (int):
            BT, row-wise block shape
        BS (int):
            BS, column wise block shape
        NAtS_Block_Size (int):
            nats chunk size
        attn_offset (int):
            attention offset
        n_data_in_current_chunk (int):
            number of data in the current chunk.
        sliding_window_size (int | None):
            sliding window size

    Returns:

    """

    BATCH, _, HAttns, N_OPTs = nats_chunk_indices.shape
    n_nats_block_per_n = triton.cdiv(BS, NAtS_Block_Size)
    if BT > NAtS_Block_Size:
        attN_idx_max_in_chunks = nats_chunk_indices[..., attn_offset].view(BATCH, -1, n_nats_block_per_n, HAttns)[:, :,
                                 -1, :]
    else:
        attN_idx_max_in_chunks = nats_chunk_indices[..., attn_offset]
    m_start = torch.arange(triton.cdiv(T, BT), device=nats_chunk_indices.device) * BT + n_data_in_current_chunk
    # If sliding window size is not None, blocks containing the last few sliding_window_size will be ignored in
    # the first iteration as  we should always query those values
    if sliding_window_size is not None:
        n_iters_per_block = (attN_idx_max_in_chunks.unsqueeze(1) * NAtS_Block_Size + sliding_window_size < m_start.view(1, -1, 1, 1)).sum(2)
    else:
        n_iters_per_block = (attN_idx_max_in_chunks.unsqueeze(1) * NAtS_Block_Size < m_start.view(1, -1, 1, 1)).sum(2)
    return n_iters_per_block


def compute_starting_idx_for_chunks(
        nats_block_indices: torch.Tensor,
        T: int,
        BT: int,
        NAtS_Block_Size: int,
        offset_op: int
):
    """
    This function is implemented to check what is the maximal last nats chunk (ordered in nats_chunk_indices)
     where all the indices values are smaller than the starting points from each computation chunk. This function is
     implemented to align the discrepancy between the nats blocks and the computational blocks
     This funciton is nearly the same as `compute_attn_n_iters_per_block`, however, we put them here to make
     it clearer on its usage TODO merge this with compute_attn_n_iters_per_block
    Args:
        nats_block_indices(torch.Tensor):
            a torch tensor of shape [B, HNAtS, T_NAtS], indicating which columns are attention columns
        T (int):
            input context length
        BT (int):
            BT, chunk sizes
        NAtS_Block_Size (int):
            nats chunk size
        offset_op (int):
            operation offset

    Returns:

    """
    BATCH, _, HAttns, N_OPTs = nats_block_indices.shape
    n_nats_block_per_n = triton.cdiv(BT, NAtS_Block_Size)
    if BT > NAtS_Block_Size:
        attN_idx_max_in_chunks = nats_block_indices[..., offset_op].view(BATCH, -1, n_nats_block_per_n, HAttns)[:, :,
                                 -1, :]
    else:
        attN_idx_max_in_chunks = nats_block_indices[..., offset_op]
    m_start = torch.arange(triton.cdiv(T, BT), device=nats_block_indices.device) * BT

    n_iters_per_block = (attN_idx_max_in_chunks.unsqueeze(1) * NAtS_Block_Size < m_start.view(1, -1, 1, 1)).sum(2)
    return n_iters_per_block