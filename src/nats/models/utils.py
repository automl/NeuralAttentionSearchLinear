# -*- coding: utf-8 -*-
import copy
from dataclasses import dataclass
from einops import rearrange
import torch
from torch.nn import functional as F
import triton
import transformers

from typing import Any, Dict, List, Optional, Tuple

from fla.models.utils import Cache

import inspect
from typing import Any, Dict, List, Optional, Union

import torch
import transformers
from packaging import version
from transformers.cache_utils import Cache as HFCacheBase
from transformers.generation import GenerationMixin
from transformers.utils.deprecation import deprecate_kwarg

_TF_VERSION = transformers.__version__
_NEED_NEW = "4.53.3"
_IS_TRANSFORMERS_4_56_PLUS = version.parse(_TF_VERSION) >= version.parse("4.56.0")

if version.parse(_TF_VERSION) > version.parse(_NEED_NEW):
    from transformers.cache_utils import CacheLayerMixin
else:
    CacheLayerMixin = object


class NAtSLayerCache(CacheLayerMixin):
    is_compileable = True

    def __init__(self,
                 seen_tokens: int = 0,
                 n_tokens_in_nats_block: int = 0,
                 n_attn_blocks: [torch.Tensor] = None,
                 nats_block_size: int = 64,
                 attn_offset: int = 0,
                 recurrent_offset: int = 1,
                 n_ops: int = 2,
                 op_for_incomplete_chunk: str = 'attn'
                 ) -> 'NAtSLayerCache':
        super().__init__()
        self.seen_tokens = seen_tokens

        self.attn_k: Optional[torch.Tensor] = None
        self.attn_v: Optional[torch.Tensor] = None
        self.l_attn_k: Optional[torch.Tensor] = None
        self.l_attn_v: Optional[torch.Tensor] = None
        self.recurrent_state: Optional[torch.Tensor] = None
        self.recurrent_state_block_start: Optional[torch.Tensor] = None
        self.block_type_state: Optional[torch.Tensor] = None
        self.conv_state = None
        self.ffn_state = None
        # TODO the current implementation only involves the cases where all the sequence in the batch starts with the
        # same opearionts
        self._n_tokens_in_nats_block = n_tokens_in_nats_block
        self.nats_block_size = nats_block_size
        self.block_type_state_logits = None

        self.n_attn_blocks = n_attn_blocks  # we note that here only the complete blocks are considered!!!
        self.n_attn_blocks_max = 0
        self.n_attn_tokens_max = 0
        self.n_ops = n_ops

        self.attn_offset = attn_offset
        self.recurrent_offset = recurrent_offset
        self.n_groups_nats = None
        self.valid_attn_tokens = None
        self.op_for_incomplete_chunk = op_for_incomplete_chunk
        self.g_cumsum = None
        # TODO check if we actually need n_nats_blocks or something else!!!
        self.n_observed_tokens = 0

    def update_attn_types(self, input_hidden_states: torch.Tensor) -> Optional[torch.Tensor]:
        B, T, D = input_hidden_states.shape

        if self.block_type_state is None:
            self.block_type_state = torch.zeros([B, self.nats_block_size, D],
                                                dtype=input_hidden_states.dtype, device=input_hidden_states.device)
        # we store the last few tokens such that they could be used in the next iteration
        n_tokens_all = T + self._n_tokens_in_nats_block
        if n_tokens_all >= self.nats_block_size:
            # we remove all the tokens in the current hidden states
            new_input_hidden_states = torch.cat(
                [self.block_type_state[:, :self._n_tokens_in_nats_block], input_hidden_states], dim=1
            )

            self.block_type_state.zero_()
            n_new_tokens = n_tokens_all % self.nats_block_size
            if n_new_tokens > 0:
                self.block_type_state[:, :n_new_tokens] = new_input_hidden_states[:, -n_new_tokens:]
            # we only need the hidden states in this case
            return new_input_hidden_states
        else:
            self.block_type_state[:, self._n_tokens_in_nats_block: n_tokens_all] = input_hidden_states
            return None

    def generate_msk(self, n_data:int=1, nats_block_types: Optional[torch.Tensor] = None,
                     compute_incomplete_chunk: bool=False,
                     ):
        if n_data == 1:
            msk = self.valid_attn_tokens[:, :self.n_attn_blocks_max + 1]
            msk = torch.repeat_interleave(
                msk, self.nats_block_size, 1
            )[:, :self.n_attn_tokens_max + 1].transpose(1,2).unsqueeze(-2).bool()
        else:
            # TODO masks for incomplete chunnks
            raise NotImplementedError

        return msk

    def update_attn_cache(self,
                          attn_state: Tuple[torch.Tensor],
                          nats_block_types: torch.Tensor,
                          ):
        k_state, v_state = attn_state
        new_data_len = k_state.shape[1]

        if self.attn_k is None:
            self.attn_k = k_state
            self.attn_v = v_state
            n_new_tokens = triton.cdiv(
                new_data_len, self.nats_block_size
            ) * self.nats_block_size - new_data_len
            # We always keep the attention caches the multiple of nats_block_sizes?
            self.attn_k = F.pad(self.attn_k, (0, 0, 0, 0, 0, n_new_tokens), mode='constant', value=0)
            self.attn_v = F.pad(self.attn_v, (0, 0, 0, 0, 0, n_new_tokens), mode='constant', value=0)
            n_valid_tokens = triton.cdiv(new_data_len, self.nats_block_size)
            self.valid_attn_tokens = torch.ones([k_state.shape[0], n_valid_tokens, nats_block_types.shape[2]],
                                                 device=self.attn_k.device, dtype=torch.bool)
        else:
            if self.attn_k.shape[1] - self.n_attn_tokens_max < new_data_len:
                n_new_tokens = triton.cdiv(
                    new_data_len + self.n_attn_tokens_max - self.attn_k.shape[1], self.nats_block_size
                ) * self.nats_block_size
                self.attn_k = F.pad(self.attn_k, (0, 0, 0, 0, 0, n_new_tokens), mode='constant', value=0)
                self.attn_v = F.pad(self.attn_v, (0, 0, 0, 0, 0, n_new_tokens), mode='constant', value=0)

            self.attn_k[:, self.n_attn_tokens_max: self.n_attn_tokens_max + new_data_len] = k_state
            self.attn_v[:, self.n_attn_tokens_max: self.n_attn_tokens_max + new_data_len] = v_state
            n_new_blocks = triton.cdiv(new_data_len + self.n_attn_tokens_max - self.valid_attn_tokens.shape[1] * self.nats_block_size,
                                       self.nats_block_size
                                       )
            if n_new_blocks > 0:
                self.valid_attn_tokens = F.pad(self.valid_attn_tokens,
                                               (0, 0, 0, n_new_blocks),
                                               mode='constant', value=1)
            if self._n_tokens_in_nats_block == 0:
                # new block should be 1
                self.valid_attn_tokens[:, self.n_attn_blocks_max] = 1

        return self.attn_k[:, :self.n_attn_tokens_max + new_data_len], self.attn_v[:, :self.n_attn_tokens_max + new_data_len]

    def update_lattn_cache(self, lattn_state: Tuple[torch.Tensor],
                           mode: str,
                           ops_for_incomplete_chunks: str):
        lattn_k, lattn_v = lattn_state
        T_lattn = lattn_k.shape[1]
        if self.l_attn_k is None:
            self.l_attn_k = torch.zeros(
                lattn_k.shape[0], self.nats_block_size, lattn_k.shape[2], lattn_k.shape[3],
                device=lattn_k.device, dtype=lattn_k.dtype
            )
            self.l_attn_v = torch.zeros(
                lattn_k.shape[0], self.nats_block_size, lattn_k.shape[2], lattn_v.shape[3],
                device=lattn_k.device, dtype=lattn_k.dtype
            )
            n_tokens_in_lattn_cache = T_lattn % self.nats_block_size
            if n_tokens_in_lattn_cache == 0:
                return lattn_state
            if T_lattn > self.nats_block_size:
                T_remains = T_lattn % self.nats_block_size
                self.l_attn_k[:, :T_remains] = lattn_k[:, -T_remains:]
                self.l_attn_v[:, :T_remains] = lattn_v[:, -T_remains:]
                return lattn_k, lattn_v
            else:
                self.l_attn_k[:, :T_lattn] = lattn_k
                self.l_attn_v[:, :T_lattn] = lattn_v
                return lattn_k, lattn_v
        else:
            T_new = T_lattn + self._n_tokens_in_nats_block
            if T_new > self.nats_block_size:
                # we only return the lattn_k and v for the multiple of self.nats_block_size
                T_remains = T_new % self.nats_block_size
                self.l_attn_k[:, :T_remains] = lattn_k[:, -T_remains:]
                self.l_attn_v[:, :T_remains] = lattn_v[:, -T_remains:]
                if mode == 'chunk' or ops_for_incomplete_chunks != 'attn':
                    # THIS is required when we want a chunk form that needs to start from the chunk start #TODO merge this into chunk kernel!
                    lattn_k_new = torch.cat(
                        [self.l_attn_k[:, :self._n_tokens_in_nats_block], lattn_k],
                        dim=1
                    )
                    lattn_v_new = torch.cat(
                        [self.l_attn_v[:, :self._n_tokens_in_nats_block], lattn_v],
                        dim=1
                    )
                    return lattn_k_new, lattn_v_new
                else:
                    return lattn_k, lattn_v
            else:
                t_start = self._n_tokens_in_nats_block
                t_end = T_new
                self.l_attn_k[:, t_start:t_end] = lattn_k
                self.l_attn_v[:, t_start:t_end] = lattn_v
                return lattn_k, lattn_v

    def update(self,
               recurrent_state: Optional[Tuple[torch.Tensor]] = None,
               recurrent_state_block_start: Optional[Tuple[torch.Tensor]] = None,
               attn_state: Optional[Tuple[torch.Tensor]] = None,
               conv_state: Optional[Tuple[torch.Tensor]] = None,
               nats_block_types: Optional[torch.Tensor] = None,
               ffn_state: Optional[Tuple[torch.Tensor]] = None,
               l_attn_state: Optional[Tuple[torch.Tensor]] = None,
               g: Optional[torch.Tensor] = None,
               n_new_tokens: Optional[int] = 0,
               cache_kwargs: Optional[Dict[str, Any]] = None,
               ):
        if self.n_groups_nats is None:
            k_state = attn_state[0]
            nhead_attn = k_state.shape[2]
            n_heads_nats = nats_block_types.shape[2]
            self.n_groups_nats = nhead_attn // n_heads_nats

        if n_new_tokens + self._n_tokens_in_nats_block >= self.nats_block_size:
            # In this case, we need to update new information and remove the unnecessary attention caches
            self.recurrent_state = recurrent_state
            self.recurrent_state_block_start = recurrent_state_block_start

            attn_blocks = nats_block_types[..., 0].int()  # B, T, H
            # we now need to select the values with the corresponding KVs
            has_incomplete_blocks = (n_new_tokens + self._n_tokens_in_nats_block) % self.nats_block_size != 0
            if has_incomplete_blocks:
                # The last one should still be kept as the last block
                attn_blocks[:, -1] = 0

            # We now want to only preserve the KV caches that are mapped as
            n_blocks_per_seq = attn_blocks.sum(1).flatten()
            max_block_size = torch.max(n_blocks_per_seq)
            max_block_size_ranges = torch.arange(max_block_size, device=attn_blocks.device)
            mask = (max_block_size_ranges.unsqueeze(0) < n_blocks_per_seq.unsqueeze(1))
            shift_idx = max_block_size_ranges.repeat(len(n_blocks_per_seq))[mask.flatten()]
            non_zero_idx = torch.nonzero(attn_blocks.transpose(1,
                                                               2))  # we need to keep the form as [B, H, N] such that the sequenth length dimension can be correctly ordered!

            b_idx = non_zero_idx[:, 0]
            h_idx = non_zero_idx[:, 1]
            if self.n_attn_blocks is not None:
                new_blocks_base = self.n_attn_blocks[b_idx, h_idx]

                shift_idx += new_blocks_base
                new_kv_start_idx_block = self.n_attn_blocks.min()
                new_kv_end_idx_block = nats_block_types.shape[1] + self.n_attn_blocks_max
            else:
                new_kv_start_idx_block = 0
                new_kv_end_idx_block = nats_block_types.shape[1]
                # shift_idx = torch.cat(non_zero_idx[:, :2], shift_idx[:, None], dim=-1)
            if has_incomplete_blocks:
                incomplete_block_shift_idx = shift_idx.max() + 1
            update_idx = non_zero_idx[:, 2] + self.n_attn_blocks_max - new_kv_start_idx_block

            for kv_cache in (self.attn_k, self.attn_v):
                kv_cache = rearrange(kv_cache, 'b (t1 t2) (h1 h2) d -> b t1 t2 h1 h2 d',
                                     t2=self.nats_block_size, h2=self.n_groups_nats
                                     )
                new_data = kv_cache[:, new_kv_start_idx_block: new_kv_end_idx_block]

                fill_value = new_data[b_idx, update_idx, :, h_idx].clone()
                kv_cache[b_idx, shift_idx, :, h_idx] = fill_value
                if has_incomplete_blocks:
                    fill_value = new_data[:, -1, :].clone()
                    kv_cache[:, incomplete_block_shift_idx, :] = fill_value
                kv_cache = rearrange(kv_cache, 'b t1 t2 h1 h2 d -> b (t1 t2) (h1 h2) d',
                                     t2=self.nats_block_size, h2=self.n_groups_nats
                                     )
            n_new_blocks = torch.sum(attn_blocks, 1)
            n_last_attn_token = shift_idx.max() + 1

            if self.n_attn_blocks is None:
                n_attn_blocks_updated = n_new_blocks - new_kv_start_idx_block

            else:
                n_attn_blocks_updated = n_new_blocks + self.n_attn_blocks - new_kv_start_idx_block

            new_blocks_are_valid = torch.arange(
                n_last_attn_token - new_kv_start_idx_block, device=n_new_blocks.device
            ).view(1, n_last_attn_token - new_kv_start_idx_block, 1) < n_attn_blocks_updated.unsqueeze(1)

            self.valid_attn_tokens[:, new_kv_start_idx_block:n_last_attn_token, ] = new_blocks_are_valid

            if has_incomplete_blocks:
                self.valid_attn_tokens[:, n_last_attn_token, ] = 1
                self.valid_attn_tokens[:, n_last_attn_token + 1:, ] = 0
            else:
                self.valid_attn_tokens[:, n_last_attn_token:, ] = 0

            if self.n_attn_blocks is None:
                self.n_attn_blocks = n_new_blocks
            else:
                self.n_attn_blocks += n_new_blocks
            self.n_attn_blocks_max = self.n_attn_blocks .max()
            self._n_tokens_in_nats_block = (self._n_tokens_in_nats_block + n_new_tokens) % self.nats_block_size
            self.n_attn_tokens_max = self.n_attn_blocks_max * self.nats_block_size + self._n_tokens_in_nats_block
            if self._n_tokens_in_nats_block == 0:
                self.g_cumsum = torch.zeros(
                    g.shape[0], 1, g.shape[2], device=g.device, dtype=g.dtype
                )
            else:
                self.g_cumsum = torch.sum(g[:, -self._n_tokens_in_nats_block:], 1, keepdim=True)
        else:
            self.recurrent_state = recurrent_state
            self.recurrent_state_block_start = recurrent_state_block_start
            if self.n_attn_blocks is None:
                k_state = attn_state[0]
                n_heads_nats = nats_block_types.shape[2]

                self.n_attn_blocks = torch.zeros([k_state.shape[0], n_heads_nats],
                                                 device=k_state.device, dtype=torch.int64)
            else:
                k_state = attn_state[0]

            self._n_tokens_in_nats_block += n_new_tokens
            self.n_attn_tokens_max += n_new_tokens
            if self.g_cumsum is None:
                self.g_cumsum = torch.sum(g, 1, keepdim=True)
            else:
                self.g_cumsum += torch.sum(g, 1, keepdim=True)

        if self._n_tokens_in_nats_block == 0:
            self.block_type_state_logits = None
            # IN this case, we need to redesign the gated delta fwd for recurrent form!!!
            self.recurrent_state = self.recurrent_state_block_start
            self.recurrent_state_block_start = None  # TODO do we really need this?
        else:
            # self.n_attn_blocks = self.n_attn_blocks - self._n_tokens_in_nats_block + n_tokens_in_new_nats_block
            if self.valid_attn_tokens is not None:
                self.valid_attn_tokens[:, self.n_attn_blocks.max()] = 1

        if conv_state is not None:
            self.conv_state = conv_state
        if ffn_state is not None:
            self.ffn_state = ffn_state
        self.n_observed_tokens += n_new_tokens


class NAtSCache(transformers.cache_utils.Cache):
    """
    A cache used for storing hidden states produced by flash linear attention models.

    It stores the states of each layer as the tensor of shape `[batch_size, key_dim, value_dim]`.
    """

    is_compileable = True

    def __init__(
            self,
            seen_tokens: int = 0,
    ) -> 'NAtSCache':
        super().__init__()

        self.states: List[NAtSLayerCache | Dict[str, Any]] = []

        self._seen_tokens = seen_tokens  # Used in `generate` to keep tally of how many tokens the cache has seen

    def __getitem__(self, layer_idx: int) -> NAtSLayerCache | Dict[str, Any]:
        if layer_idx < len(self):
            return self.states[layer_idx]
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        for state in self.states:
            yield state

    def __len__(self):
        return len(self.states)

    def add_natsl_layer_cache(self,
                              seen_tokens: int =0,
                              n_tokens_in_nats_block: int=0,
                              n_attn_blocks: [torch.Tensor] = None,
                              nats_block_size: int=64,
                              attn_offset: int=0,
                              recurrent_offset: int =1,
                              n_ops:int=2,
                              op_for_incomplete_chunk: str= 'all'
                              ):
        self.states.append(
            NAtSLayerCache(seen_tokens,
                           n_tokens_in_nats_block,
                           n_attn_blocks,
                           nats_block_size,
                           attn_offset,
                           recurrent_offset,
                           n_ops,
                           op_for_incomplete_chunk))

    def update(
            self,
            recurrent_state: Optional[Tuple[torch.Tensor]] = None,
            recurrent_state_block_start: Optional[tuple[torch.Tensor]] = None,
            attn_state: Optional[Tuple[torch.Tensor]] = None,
            conv_state: Optional[Tuple[torch.Tensor]] = None,
            nats_block_types: Optional[torch.Tensor] = None,
            block_type_state: Optional[Tuple[torch.Tensor]] = None,
            ffn_state: Optional[Tuple[torch.Tensor]] = None,
            layer_idx: int = 0,
            offset: Optional[int] = 1,
            cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Args:
            recurrent_state (`torch.Tensor`):
                The new recurrent state to cache.
            attn_state (`Tuple[torch.Tensor]`):
                The new attention key/value states to cache.
            conv_state (`Tuple[torch.Tensor]`):
                The new convolution state to cache.
            ffn_state (`Tuple[torch.Tensor]`):
                The new feed-forward state to cache.
            layer_idx (`int`, defaults to 0):
                The index of the layer to cache the states for.
            offset (`int`, defaults to 1):
                The number of new tokens being processed.
            cache_kwargs (`Dict[str, Any]`):
                Additional arguments for the cache subclass.

        Return:
            Dictionary of the updated state.
        """

        if cache_kwargs is None:
            cache_kwargs = {}
        if attn_state is not None:
            input_size = attn_state[0].shape[1]
            window_size = cache_kwargs.get('window_size', None)
            if not (isinstance(attn_state, Tuple) or isinstance(attn_state, List)):
                raise ValueError("`attn_state` must be a tuple of tensors for key/value states")
        if len(self.states) <= layer_idx:
            # update the number of seen tokens
            if layer_idx == 0:
                self._seen_tokens += offset
            if attn_state is not None:
                if window_size is not None and input_size > window_size:
                    attn_state = [state[:, -window_size:].contiguous() for state in attn_state]
            state = dict(
                recurrent_state=recurrent_state,
                attn_state=attn_state,
                conv_state=conv_state,
                ffn_state=ffn_state
            )
            self.states.append(state)
        else:
            # update the number of seen tokens
            if layer_idx == len(self.states) - 1:
                self._seen_tokens += offset
            state = self.states[layer_idx]
            if recurrent_state is not None:
                state['recurrent_state'] = recurrent_state
            if attn_state is not None:
                if window_size is not None and state['attn_state'][0].shape[1] == window_size:
                    for i, (old_state, new_state) in enumerate(zip(state['attn_state'], attn_state)):
                        # DO NOT allocate new memory if the cache is full
                        # roll the key/value states to the left by `input_size`
                        old_state = old_state.roll(-input_size, 1)
                        # replace the last `input_size` tokens with the new key/value states
                        old_state[:, -input_size:] = new_state
                        state['attn_state'][i] = old_state
                else:
                    attn_state = [
                        torch.cat([old_state, new_state], 1)
                        for old_state, new_state in zip(state['attn_state'], attn_state)
                    ]
                    state['attn_state'] = attn_state
            if conv_state is not None:
                state['conv_state'] = conv_state
            if ffn_state is not None:
                state['ffn_state'] = ffn_state

        return state

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        if len(self.states) <= layer_idx:
            return 0
        return self._seen_tokens

    def get_max_cache_shape(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. Cache does not have a maximum length."""
        return None

    def to_legacy_cache(self) -> Tuple:
        return tuple(self.states)

    @property
    def n_tokens_in_nats_block(self):
        return self._n_tokens_in_nats_block

    @classmethod
    @torch.compiler.disable
    def from_legacy_cache(
            cls,
            past_key_values: Optional[Tuple] = None,
            seen_tokens: int = 0
    ) -> Cache:
        """Converts a cache in the legacy cache format into an equivalent `Cache`."""

        cache = cls(seen_tokens)
        if isinstance(past_key_values, list):
            for layer_idx in range(len(past_key_values)):
                cache.states.append(past_key_values[layer_idx])
        return cache




def test_nats_cache_attn():
    B = 4
    T1 = 1078
    nats_block_size = 64
    H = 8
    HNAtS = 4
    D = 2

    cache = NAtSLayerCache()
    k1 = (torch.arange(T1) + 1).view(1, T1, 1, 1).repeat(B, 1, H, D)
    v1 = copy.deepcopy(k1)
    import math
    import triton
    Tnats1 = triton.cdiv(T1, nats_block_size)
    nats_block_type1 = torch.randn(B, Tnats1, HNAtS, 2)
    index = nats_block_type1.max(-1, keepdim=True)[1]
    nats_block_type1 = torch.zeros_like(
        nats_block_type1, memory_format=torch.legacy_contiguous_format
    ).scatter_(-1, index, 1.0)
    k1_o, v1_o = cache.update_attn_cache((k1, v1))

    cache.update(attn_state=(k1, v1), nats_block_types=nats_block_type1, n_new_tokens=T1)
    cache_k = cache.attn_k

    n_attn_blocks_max = cache.n_attn_blocks.max()
    """
    for b in range(B):
        for h1 in range(HNAtS):
            for h2 in range(H//HNAtS):
                attn_block_type = nats_block_type1[b,:,h1, 0]
                i_cache = 0
                for i_t in range(attn_block_type.shape[0]):
                    print(f"***")
                    print(b)
                    print(h1)
                    print(h2)
                    print(i_t)

                    if attn_block_type[i_t]:
                        k_current = k1[b, i_t*nats_block_size:(i_t + 1)*nats_block_size, h2 + h1 * H//HNAtS, 0]
                        k_cache = cache_k[b, i_cache * nats_block_size : (i_cache + 1) * nats_block_size,  h2 + h1 * H//HNAtS, 0]
                        valid_token = cache.valid_attn_tokens[b,  i_cache : (i_cache + 1), h1]
                        if i_t < (attn_block_type.shape[0] - 1):
                            assert (k_current - k_cache[:len(k_current)]).abs().sum() == 0
                            assert valid_token.all()
                        else:
                            k_cache =  cache_k[b, n_attn_blocks_max * nats_block_size : (n_attn_blocks_max + 1) * nats_block_size,  h2 + h1 * H//HNAtS, 0]
                            assert (k_current - k_cache[:len(k_current)]).abs().sum() == 0
                            if len(k_current) < len(k_cache):
                                assert k_cache[len(k_current):].abs().sum() == 0
                        i_cache += 1

                valid_token = cache.valid_attn_tokens[b,(i_cache + 1) * nats_block_size:, h1]
                assert valid_token.int().sum() == 0
    """

    import pdb
    pdb.set_trace()


def test_nats_cache_attn_chunk_wise():
    torch.manual_seed(0)
    B = 4
    # T1 = 1078
    T1 = 1728
    T2 = 45
    nats_block_size = 64
    H = 8
    HNAtS = 4
    D = 2

    T12 = T1 + T2
    cache1 = NAtSLayerCache()
    k12 = ((torch.arange(T1 + T2)) + 1).view(1, T1 + T2, 1, 1).repeat(B, 1, H, D)
    v12 = copy.deepcopy(k12)
    import triton
    Tnats12 = triton.cdiv(T12, nats_block_size)
    nats_block_type12 = torch.randn(B, Tnats12, HNAtS, 2)

    index = nats_block_type12.max(-1, keepdim=True)[1]
    nats_block_type12 = torch.zeros_like(
        nats_block_type12, memory_format=torch.legacy_contiguous_format
    ).scatter_(-1, index, 1.0)
    k12_o, v12_o = cache1.update_attn_cache((k12, v12))

    cache1.update(attn_state=(k12, v12), nats_block_types=nats_block_type12, n_new_tokens=T12)
    cache_k1 = cache1.attn_k

    k21 = k12[:, :T1]
    k22 = k12[:, :T2]

    v21 = v12[:, :T1]
    v22 = v12[:, :T2]

    nats_block_type21 = nats_block_type12[:, :T1 // nats_block_size]
    nats_block_type22 = nats_block_type12[:, T1 // nats_block_size:]
    if (T1 % nats_block_size) > 0:
        # the first nats block type needs to be padded
        nats_block_type21 = torch.cat(
            [nats_block_type21, torch.zeros(B, 1, HNAtS, 2).to(nats_block_type21)], dim=1
        )
    cache2 = NAtSLayerCache()
    k21_o, v21_o = cache2.update_attn_cache((k21, v21))
    cache2.update(attn_state=(k21, v21), nats_block_types=nats_block_type21, n_new_tokens=T1)

    k22_o, v2_o = cache2.update_attn_cache((k22, v22))
    cache2.update(attn_state=(k22, v22), nats_block_types=nats_block_type22, n_new_tokens=T2)
    cache_k2 = cache1.attn_k
    assert (cache_k1 - cache_k2).abs().sum() == 0
    valid_attn_tokens1 = cache1.valid_attn_tokens
    valid_attn_tokens2 = cache2.valid_attn_tokens

    assert (valid_attn_tokens1 == valid_attn_tokens2).all()

    n_attn_blocks1 = cache1.n_attn_blocks
    n_attn_blocks2 = cache2.n_attn_blocks

    assert (n_attn_blocks1 == n_attn_blocks2).all()

    n_tokens_in_nats_block1 = cache1._n_tokens_in_nats_block
    n_tokens_in_nats_block2 = cache2._n_tokens_in_nats_block
    import pdb
    pdb.set_trace()
    assert (n_tokens_in_nats_block1 == n_tokens_in_nats_block2)

    import pdb
    pdb.set_trace()


def test_nats_cache_attn_iter():
    torch.manual_seed(0)
    B = 4
    # T1 = 1078
    T1 = 1777
    T2 = 195
    nats_block_size = 64
    H = 8
    HNAtS = 4
    D = 2

    T12 = T1 + T2
    cache1 = NAtSLayerCache()
    k12 = ((torch.arange(T1 + T2)) + 1).view(1, T1 + T2, 1, 1).repeat(B, 1, H, D)
    v12 = copy.deepcopy(k12)
    import triton
    Tnats12 = triton.cdiv(T12, nats_block_size)
    nats_block_type12 = torch.randn(B, Tnats12, HNAtS, 2)

    index = nats_block_type12.max(-1, keepdim=True)[1]
    nats_block_type12 = torch.zeros_like(
        nats_block_type12, memory_format=torch.legacy_contiguous_format
    ).scatter_(-1, index, 1.0)
    k12_o, v12_o = cache1.update_attn_cache((k12, v12))

    cache1.update(attn_state=(k12, v12), nats_block_types=nats_block_type12, n_new_tokens=T12)
    cache_k1 = cache1.attn_k

    k21 = k12[:, :T1]
    k22 = k12[:, :T2]

    v21 = v12[:, :T1]
    v22 = v12[:, :T2]

    nats_block_type21 = nats_block_type12[:, :T1 // nats_block_size]
    nats_block_type22 = nats_block_type12[:, T1 // nats_block_size:]
    if (T1 % nats_block_size) > 0:
        # the first nats block type needs to be padded
        nats_block_type21 = torch.cat(
            [nats_block_type21, torch.zeros(B, 1, HNAtS, 2).to(nats_block_type21)], dim=1
        )
    cache2 = NAtSLayerCache()
    k21_o, v21_o = cache2.update_attn_cache((k21, v21))
    cache2.update(attn_state=(k21, v21), nats_block_types=nats_block_type21, n_new_tokens=T1)

    for i in range(T2):
        k22_o, v2_o = cache2.update_attn_cache((k22[:, [i]], v22[:, [i]]))
        cache2.update(attn_state=(k22[:, [i]], v22[:, [i]]),
                      nats_block_types=nats_block_type12[:, [(T1 + i) // nats_block_size]], n_new_tokens=1)
    cache_k2 = cache1.attn_k
    valid_attn_tokens1 = cache1.valid_attn_tokens
    valid_attn_tokens2 = cache2.valid_attn_tokens

    n_attn_blocks1 = cache1.n_attn_blocks
    n_attn_blocks2 = cache2.n_attn_blocks

    n_tokens_in_nats_block1 = cache1._n_tokens_in_nats_block
    n_tokens_in_nats_block2 = cache2._n_tokens_in_nats_block

    import pdb
    pdb.set_trace()

    assert (cache_k1 - cache_k2).abs().sum() == 0
    if valid_attn_tokens1 is not None:
        assert (valid_attn_tokens1 == valid_attn_tokens2).all()
    assert (n_attn_blocks1 == n_attn_blocks2).all()
    import pdb
    pdb.set_trace()
    assert (n_tokens_in_nats_block1 == n_tokens_in_nats_block2)

    import pdb
    pdb.set_trace()


if __name__ == "__main__":
    # only works on post-Ampere GPUs right now
    # test_compute_h()
    # test_nats_cache_attn()
    test_nats_cache_attn_chunk_wise()
    # test_nats_cache_attn_iter()
