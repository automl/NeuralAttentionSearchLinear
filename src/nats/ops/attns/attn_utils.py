import torch

def repeat_masks(mask: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This function is similar to repeat_kv, however, we will expand the second dimension
    """
    bs, n_kv_heads, slen1, slen2 = mask.shape
    if n_rep == 1 or n_kv_heads == 1:
        return mask
    return (
        mask[:, :, None, :, :]
            .expand(bs, n_kv_heads, n_rep, slen1, slen2)
            .reshape(bs, n_kv_heads * n_rep, slen1, slen2)
    )