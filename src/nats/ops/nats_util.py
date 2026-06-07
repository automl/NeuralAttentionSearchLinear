import triton
import torch
from torch.nn import functional as F


def prepare_nats_block_indices(
    n_nats_blocks: torch.LongTensor,
    nats_block_size: int,
    chunk_size: int,
) -> torch.LongTensor:
    """
    prepeare the chunk indices for nats. We notice that since the nats_block_size might be different for different nats
    heads, instead of returning a tensor of shape [n_blocks, 2] where the first value indicates the batch indices and
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
        indices: torch.Tensor
            chunk indices for nats.

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
    prepare the nats chunk offsets. Since we do not need to compute the chunk-wise hidden states for non-target operation blocks, we consider the
    chunks with the corresponding op_idx as a sequence with variable length. We use nats_chunk_offsets to record how many  chunks we need to compute 
    within each sequence for the target opeartion
    Args:
        n_nats_blocks: torch.Tensor, 
            number of nats chunk for each operation, a tensor of shape [B, HNAtS, n_opts]
        nats_block_types: torch.Tensor
            nats chunk block types, a tensor of shape [B, TNAtS, HNAtS, n_opts]
        nats_block_size: int
            nats block size
        chunk_size: int
            linear attention computing chunk size
        op_idx: torch.Tensor
            index of the target operation

    Returns:
        nats_chunk_offsets:
            the offset for each sequence for the target operation. 

    """
    n_chunks = triton.cdiv(n_nats_blocks[..., op_idx] * nats_block_size, chunk_size).flatten()
    return F.pad(n_chunks, (1,0), mode='constant', value=0).cumsum(0)


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
    This function is used to compute how many full iterations we need to compute for each softmax attention sequence.
    We note that here full iteration indicates that every element loaded to SRAM will be computed and no masking is applied. 
    So assuming that we have a nats_block_size = 32 and BS = 64, where chunks [0,1,3] is considered as softmax attention chunks. Then the iteration
    number is 1 (for chunk 0,1) because only chunk 3 will not occupy an entire computational block
    Args:
        nats_chunk_indices: torch.Tensor,
            a torch tensor of shape [B, HNAtS, T_NAtS], indicating which columns are attention columns
        T: int,
            input context length
        BT: int,
            BT: flash attention computational block size (row-wise), how many Q vlaues do we load into SRAM
        BS: int,
            BS: flash attention computational block size (column-wise), how many K values do we load into SRAM
        NAtS_Block_Size: int,
            nats chunk size
        attn_offset: int,
            softmax attention offset
        n_data_in_current_chunk: int,
            number of data in the current chunk. This can be used for continuously pre-filling, there might be some historical tokens that cannot fit in 
            a complete chunk. Therefore, they need to wait to form chunks with the current iteartion. And we need to take these tokens into consideration.
        sliding_window_size: int | None,
            sliding window size. If this value is set any positive value, then every token will be computed with at least the previous sliding_window_size 
            tokens when doing softmax attention opeartion. i.e., even for the first token of a softmax attention chunk with its previous chunk being
            linear attention chunk, the token will still include the last sliding_window_size tokens even they belong to linear attention chunks.

    Returns:
        n_iters_per_block: torch.Tensor,
            a tensor of shape [BT, T//BT, T//NAtS_BlockSize, HNAtS]
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
        n_iters_per_block = ((attN_idx_max_in_chunks.unsqueeze(1) + 1) * NAtS_Block_Size <= m_start.view(1, -1, 1, 1)).sum(2)
    return n_iters_per_block


def compute_starting_idx_for_chunks(
        nats_block_indices: torch.Tensor,
        T: int,
        BT: int,
        NAtS_Block_Size: int,
        offset_op: int
):
    """
    This function is implemented to check what is the maximal last nats chunk (ordered in nats_chunk_indices). This function is implemented to 
    tell which hidden state the current chunk should read when we compute the linear attention output values.
    For instance, if we have 8 chunks and chunks [2, 5] are linear attetnion chunks. Then the result should be [0,0, 2, 2, 2, 5, 5, 5]
    Args:
        nats_block_indices: torch.Tensor,
            a torch tensor of shape [B, TNAtS, HNAtS, N_OPTs], indicating which columns are attention columns
        T: int,
            input context length
        BT: int,
            BT, chunk sizes
        NAtS_Block_Size: int,
            nats chunk size
        offset_op: int),
            operation offset

    Returns:
        starting_idx: torch.Tensor,
            starting index of each chunk, of shape [B, T//BT, TNAtS, HNAtS]

    """
    BATCH, _, HAttns, N_OPTs = nats_block_indices.shape
    n_nats_block_per_n = triton.cdiv(BT, NAtS_Block_Size)
    if BT > NAtS_Block_Size:
        attN_idx_max_in_chunks = nats_block_indices[..., offset_op].view(BATCH, -1, n_nats_block_per_n, HAttns)[:, :,
                                 -1, :]
    else:
        attN_idx_max_in_chunks = nats_block_indices[..., offset_op]
    m_start = torch.arange(triton.cdiv(T, BT), device=nats_block_indices.device) * BT

    starting_idx = (attN_idx_max_in_chunks.unsqueeze(1) * NAtS_Block_Size < m_start.view(1, -1, 1, 1)).sum(2)
    return starting_idx