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


class NAtSLayerCache:
    is_compileable = True

    def __init__(self,
                 seen_tokens: int = 0,
                 n_tokens_in_nats_block: int = 0,
                 n_attn_blocks: torch.Tensor | None = None,
                 nats_block_size: int = 64,
                 attn_offset: int = 0,
                 recurrent_offset: int = 1,
                 n_ops: int = 2,
                 op_for_incomplete_chunk: str = 'attn'
                 ) -> 'NAtSLayerCache':
        """
        A NAtS Layer Cache. This class is implemented to manage all the caches implemented in NAtSL. Depending on which operation we use for 
        incomplete chunks, NAtSL stores and rerolls its cache in different ways. 
        If op_for_incomplete_chunk is 'all', both attn and linear attn caches will be updated at each iteartion and we remove new KV attn values if the new 
        chunk is a linear attention chunk or reroll the linear attetnion's KV values if the new chunk is a softmax attetnion chunk. Hence, in addition to softmax 
        KV cache, linear attention hidden state at the current time step, we also needs to store linear attention's hidden state at the start of the chunk.
        Additionally, since we need to ensure that the linear attention chunks align with the NAtS chunks when we do contiual pre-filling with some tokens 
        remaining in the last chunk, we cannot directly start another linear attention parallel forward process. Hence, we first concatnate the old linear attention
        related information (l_attn_k, l_attn_v, lattn_g, lattn_delta) to the corresponding values of the new tokens and then run parallel forward pass with 
        the concatnated information. Additionally, if the intra-chunk operation does not involve linear attention, we also need to record all the linear attention 
        related information. Since we no longer compute the intermedaite hidden state at each time step but only update an entire chunk with parallel form at the 
        end of each chunk.


        Attributes:
            attn_k: torch.Tensor | None,
                Attention K cache
            attn_v: torch.Tensor | None,
                Attention V cache
            l_attn_k: torch.Tensor | None,
                Linear Attention K cache, with a shape of [batch_size, NAtS_chunk_size, n_heads, head_dims_k], 
                used to concatate with the new linear attention K tokens for parallel chunking computation. 
            l_attn_v: torch.Tensor | None,
                Linear Attention V cache, with a shape of [batch_size, NAtS_chunk_size, n_heads, head_dims_k],
                used to concatate with the new linear attention V tokens for parallel chunking computation. 
            lattn_g: torch.Tensor | None,
                gate value of linear attentionm with a shape of [batch_size, NAtS_chunk_size, n_heads], 
                used to concatate with the new linear attention g tokens for parallel chunking computation. 
            lattn_beta: torch.Tensor | None,
                beta value of linear attention with a shape of [batch_size, NAtS_chunk_size, n_heads],
                used to concatate with the new linear attention beta tokens for parallel chunking computation. 
            recurrent_state: torch.Tensor | None,
                linear attention recurrent state at the current stage. This value is not requried if the operations for incomplete chunks does not contain linear attention 
            recurrent_state_block_start: torch.Tensor | None,
                linear attention recurrent state in the beginning of the current chunk. This value is stored to ensure that the recurrent_state can be restored if the new chunk
                is not a linear attention chunk.
            block_type_state: torch.Tensor,
                this value stored the input feature maps that used to compute the attention scores with a shape of [batch_size, NAtS_chunk_size, model_dim]. Here we store the entire 
                feature map in the chunk. We could also only store the current mean values of the feature maps. However, to ensure that the model provides consistent results, we still 
                store the entire chunk here.
            conv_state: torch.Tensor,
                The new convolutional state to cache.
            ffn_state: torch.Tenspr,
                The new feed-forward state to cache.
            nats_block_size: int,
                nats block sizes
            n_attn_blocks_max: int,
                maximal number of softmax attention chunks. We note that this value only contain the complete chunks that are already classified as softmax attention chunks. This
                value helps us to extract the KV cache values for softmax attention
            n_attn_tokens_max: int,
                maximal number of softmax attention tokens. This can be computed as n_attn_blocks_max * nats_chunk_size + n_tokens_in_new_chunk
            n_ops: int,
                number of operations in our search space
            attn_offset: int,
                offset for softmax attentions
            recurrent_offset: int,
                offset for linear attention operations 
            valid_attn_tokens: torch.Tensor
                Since different batches (or different heads within the same batch) might contains different number of softmax attention chunks, we need to record these values to show 
                which tokens are actually softmax attention tokens. This can then be directly applied as an attention mask for softmax attention operations 
            op_for_incomplete_chunk: str,
                operations for incomplete chunks, currently, we have 'all': both linear attention and softmax attentions are involved, 'attn': only softmax attention is considered, 
                'gated_delta_net', only gated delta net (or linear attention is involved)
            g_cumsum: torch.Tenspr,
                cumulative sum of the g values within the current block. Since we might need to decay the hidden states regardless of the current chunk is a lienar attention chunk 
                or softmax attention tokens, we compute cumulative sum of g to decay properly. 
            n_observed_tokens: int,
                number of obsered tokens for this cache, used for positional encoding. 
            
        """
        super().__init__()
        self.seen_tokens = seen_tokens

        self.attn_k: Optional[torch.Tensor] = None
        self.attn_v: Optional[torch.Tensor] = None
        self.l_attn_k: Optional[torch.Tensor] = None
        self.l_attn_v: Optional[torch.Tensor] = None
        self.lattn_g :Optional[torch.Tensor] = None
        self.lattn_beta: Optional[torch.Tensor] = None
        self.recurrent_state: Optional[torch.Tensor] = None
        self.recurrent_state_block_start: Optional[torch.Tensor] = None
        self.block_type_state: Optional[torch.Tensor] = None
        self.conv_state = None
        self.ffn_state = None
        # TODO the current implementation only involves the cases where all the sequence in the batch starts with the
        # same amount of tokens in new chunk size. In which case, _n_tokens_in_nats_block should be a tensor instead of a scalar value
        self._n_tokens_in_nats_block = n_tokens_in_nats_block # Number of tokens in the most recent chunk (value between (0, nats_block_size))
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

    def lazy_initialization(self, key_states: torch.Tensor):
        """
        initialize all the hidden states into initial states
        """
        self.seen_tokens = 0

        self.attn_k: Optional[torch.Tensor] = None
        self.attn_v: Optional[torch.Tensor] = None
        self.l_attn_k: Optional[torch.Tensor] = None
        self.l_attn_v: Optional[torch.Tensor] = None
        self.recurrent_state: Optional[torch.Tensor] = None
        self.recurrent_state_block_start: Optional[torch.Tensor] = None
        self.block_type_state: Optional[torch.Tensor] = None
        self.conv_state = None
        self.ffn_state = None
        self._n_tokens_in_nats_block = 0
        self.block_type_state_logits = None

        self.n_attn_blocks = None  # we note that here only the complete blocks are considered!!!
        self.n_attn_blocks_max = 0
        self.n_attn_tokens_max = 0

        self.n_groups_nats = None
        self.valid_attn_tokens = None
        self.g_cumsum = None
        self.n_observed_tokens = 0

    def update_attn_types(self, input_hidden_states: torch.Tensor) -> Optional[torch.Tensor]:
        """
        update the hidden states that can be later applied to check the chunk types

        Args:
            input_hidden_states: torch.Tensor
                input hidden states, i.e., the archtecture input. 
        """
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
        """
        generate a mask used by softmax attention operations. If a KV cache contains sequence with different length, 
        we put all tokens of incomplete chunks to the end of the last complete softmax chunks. For instance, if we have
        an observed sequence of 
        [[sa, sa, new_token],
         [sa, la, new_token]]
        then the corresponding mask should be:
        [[T, T, T],
         [T, F, T]]
        We need to call this function after updaing the softmax KV cache values. 
         
        Args:
            n_data: int
                data sequence lenth of new input data
            nats_block_types: torch.Tensor
                the types of the new chunk blocks, this can be applied if n_data + n_token_in_chunks is larger than 
                the chunk size
            compute_incomplete_chunk: bool
                if we would like to compute intra-chunk correlation with softmax attention. This item determines if we want 
                to set the incomplete chunks as True or False.
            
        """
        if n_data == 1:
            msk = self.valid_attn_tokens[:, :self.n_attn_blocks_max + 1]
            msk = torch.repeat_interleave(
                msk, self.nats_block_size, 1
            )

            if self.op_for_incomplete_chunk != 'gated_delta_net':
                num_msk_columns = self.n_attn_tokens_max + 1
            else:
                num_msk_columns = (self.n_attn_tokens_max + 1) // self.nats_block_size * self.nats_block_size
            msk = msk[:, :num_msk_columns].transpose(1,2).unsqueeze(-2).bool()
        else:
            if nats_block_types.shape[1] == 1:
                # in this case, we do normal causal mask
                if compute_incomplete_chunk:
                    msk = torch.ones(n_data, n_data + self._n_tokens_in_nats_block, device=nats_block_types.device,
                                     dtype=nats_block_types.dtype)
                    msk = msk.tril_(self._n_tokens_in_nats_block)
                else:
                    # TODO how to sol;ve this?
                    msk = torch.zeros(n_data, n_data + self._n_tokens_in_nats_block, device=nats_block_types.device,
                                     dtype=nats_block_types.dtype)
                msk = msk.view(1,1, n_data, n_data + self._n_tokens_in_nats_block).bool()
            else:
                valid_columns = torch.transpose(nats_block_types[..., self.attn_offset], 1, 2)
                mask_current = valid_columns.unsqueeze(-2)
                n_datar_cur = self._n_tokens_in_nats_block + n_data

                if compute_incomplete_chunk:
                    mask_diag = torch.ones(self.nats_block_size, self.nats_block_size,
                                           dtype=nats_block_types.dtype, device=mask_current.device).tril_(0)
                    n_datar_cur = self._n_tokens_in_nats_block + n_data
                    mask_diag = torch.block_diag(*[mask_diag for _ in range(nats_block_types.shape[1])])[self._n_tokens_in_nats_block:n_datar_cur, :n_datar_cur]
                    mask_current = mask_current.repeat_interleave(self.nats_block_size, 3)[..., :n_datar_cur]
                    mask_current = mask_diag[None, None, :, :] + mask_current
                    mask_current = mask_current.tril_(self._n_tokens_in_nats_block)
                else:
                    n_new_chunks = nats_block_types.shape[1]
                    full_chunks = torch.ones(n_new_chunks,n_new_chunks,
                                             device=mask_current.device,
                                             dtype=mask_current.dtype).tril(-1)
                    full_chunks = full_chunks.repeat_interleave(
                        self.nats_block_size, 0).repeat_interleave(self.nats_block_size, 1)
                    mask_current = (valid_columns.unsqueeze(-2) * full_chunks[None, None, :, :])[self._n_tokens_in_nats_block:n_datar_cur, :n_datar_cur]

                if self.n_attn_blocks_max > 0:
                    mask_past = self.valid_attn_tokens[:, :self.n_attn_blocks_max]
                    mask_past = torch.repeat_interleave(
                        mask_past, self.nats_block_size, 1
                    ).transpose(1, 2).unsqueeze(-2)
                    msk = torch.cat([mask_past, mask_current], dim=-1).bool()
                else:
                    msk = mask_current
        return msk

    def update_attn_cache(self,
                          attn_state: Tuple[torch.Tensor],
                          nats_block_types: torch.Tensor,
                          ):
        """
        Update softmax attention KV cache given the types of new nats chunks

        Args:
            attn_states: Tuple[torch.Tensor, torch.Tensor],
                softmax attention states, storing the K, V values of current tokens accordingly
            nats_block_types: torch.Tensor,
                attention types of new token chunk
        """
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
            return k_state, v_state
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

        if self.op_for_incomplete_chunk != 'gated_delta_net':
            return self.attn_k[:, :self.n_attn_tokens_max + new_data_len], self.attn_v[:, :self.n_attn_tokens_max + new_data_len]
        else:
            n_tokens = (self.n_attn_tokens_max + new_data_len) // self.nats_block_size * self.nats_block_size
            return self.attn_k[:, :n_tokens], self.attn_v[:, :n_tokens]

    def update_lattn_cache(self,
                           lattn_state: Tuple[torch.Tensor],
                           lattn_beta: torch.Tensor,
                           lattn_g: Optional[torch.Tensor],
                           mode: str,
                           ops_for_incomplete_chunks: str):
        """
        Update linear attention realted states given the types of new nats chunks. We need to record intermediate beta
        and g values in case we want to do coninuing pre-filling, these intermediate values are applied to do parallel forward
        with hidden_state_chunk_start. Hence, all the values of the current incomplete chunks are stored

        Args:
            lattn_state: Tuple[torch.Tensor, torch.Tensor],
                Linear attention states, storing the K, V values of current tokens accordingly
            lattn_beta: torch.Tensor,
                beta value of the linear attention
            lattn_g: torch.Tensor,
                decay of the linaer attention
            mode: str,
                forward mode. It can be parallel or recurrent
            ops_for_incomplete_chunks: str, 
                operations for incomplete chunks. This can be 'attn', 'gated_delta_net', or 'all'
        """
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
            if lattn_beta is not None:
                self.lattn_beta = torch.zeros(
                    lattn_k.shape[0], self.nats_block_size, lattn_k.shape[2],
                    device=lattn_k.device, dtype=lattn_beta.dtype
                )
            if lattn_g is not None:
                self.lattn_g = torch.zeros(
                    lattn_k.shape[0], self.nats_block_size, lattn_k.shape[2],
                    device=lattn_g.device, dtype=lattn_g.dtype
                )
            n_tokens_in_lattn_cache = T_lattn % self.nats_block_size
            if n_tokens_in_lattn_cache == 0:
                return lattn_k, lattn_v, lattn_beta, lattn_g
            if T_lattn > self.nats_block_size:
                T_remains = T_lattn % self.nats_block_size
                self.l_attn_k[:, :T_remains] = lattn_k[:, -T_remains:]
                self.l_attn_v[:, :T_remains] = lattn_v[:, -T_remains:]
                if lattn_beta is not None:
                    self.lattn_beta[:,:T_remains] = lattn_beta[:, -T_remains:]
                if lattn_g is not None:
                    self.lattn_g[:,:T_remains] = lattn_g[:, -T_remains:]
                return lattn_k, lattn_v, lattn_beta, lattn_g
            else:
                self.l_attn_k[:, :T_lattn] = lattn_k
                self.l_attn_v[:, :T_lattn] = lattn_v
                if lattn_beta is not None:
                    self.lattn_beta[:, :T_lattn] = lattn_beta
                if lattn_g is not None:
                    self.lattn_g[:, :T_lattn] = lattn_g
                return lattn_k, lattn_v, lattn_beta, lattn_g
        else:
            T_new = T_lattn + self._n_tokens_in_nats_block
            if T_new >= self.nats_block_size:
                # we only return the lattn_k and v for the multiple of self.nats_block_size
                T_remains = T_new % self.nats_block_size
                self.l_attn_k[:, :T_remains] = lattn_k[:, -T_remains:]
                self.l_attn_v[:, :T_remains] = lattn_v[:, -T_remains:]
                if lattn_beta is not None:
                    self.lattn_beta[:, :T_remains] = lattn_beta[:, -T_remains:]
                if lattn_g is not None:
                    self.lattn_g[:, :T_remains] = lattn_g[:, -T_remains:]
                if mode == 'chunk' or ops_for_incomplete_chunks == 'attn':
                    # THIS is required when we want a chunk form that needs to start from the chunk start #TODO merge this into chunk kernel!
                    lattn_k_new = torch.cat(
                        [self.l_attn_k[:, :self._n_tokens_in_nats_block], lattn_k],
                        dim=1
                    )
                    lattn_v_new = torch.cat(
                        [self.l_attn_v[:, :self._n_tokens_in_nats_block], lattn_v],
                        dim=1
                    )
                    if lattn_beta is not None:
                        lattn_beta_new = torch.cat(
                            [self.lattn_beta[:, :self._n_tokens_in_nats_block], lattn_beta],
                            dim=1
                        )
                    else:
                        lattn_beta_new = None
                    if lattn_g is not None:
                        lattn_g_new = torch.cat(
                            [self.lattn_g[:, :self._n_tokens_in_nats_block], lattn_g],
                            dim=1
                        )
                    else:
                        lattn_g_new = None
                    return lattn_k_new, lattn_v_new, lattn_beta_new, lattn_g_new
                else:
                    return lattn_k, lattn_v, lattn_beta, lattn_g
            else:
                t_start = self._n_tokens_in_nats_block
                t_end = T_new
                self.l_attn_k[:, t_start:t_end] = lattn_k
                self.l_attn_v[:, t_start:t_end] = lattn_v
                if lattn_beta is not None:
                    self.lattn_beta[:, t_start:t_end] = lattn_beta
                if lattn_g is not None:
                    self.lattn_g[:, t_start:t_end] = lattn_g

                return lattn_k, lattn_v, lattn_beta, lattn_g

    def update(self,
               recurrent_state: Optional[Tuple[torch.Tensor]] = None,
               recurrent_state_block_start: Optional[Tuple[torch.Tensor]] = None,
               attn_state: Optional[Tuple[torch.Tensor]] = None,
               conv_state: Optional[Tuple[torch.Tensor]] = None,
               nats_block_types: Optional[torch.Tensor] = None,
               ffn_state: Optional[Tuple[torch.Tensor]] = None,
               g: Optional[torch.Tensor] = None,
               n_new_tokens: Optional[int] = 0,
               cache_kwargs: Optional[Dict[str, Any]] = None,
               ):
        """
        Update the hideen states of both linear and softmax attentions. Since adding new values to the caches are done within the funciton 
        update_attn_cache and update_lattn_cache. This function is called only for post-update. For instance, reroll the hidden states to the start 
        if the current token is a softmax attention token or removing new KV caches of linear attention blocks.
        Additionally, we need to fill the most recent KV cache block values to the end of the current sequecne.
        For instance, if we have the following cache arrangement:
        [[T, T, T],
         [T, F, T]]
        Since the new chunk is a softmax token, we need to move the 3rd chunk in head 2 to place 2:
        [[T, T, T],
         [T, T, F]]

        Args:
            recurrent_state: torch.Tensor,
                recurrent hidden state of linear attention at the current time step
            recurrent_state_block_start: torch.Tensor,
                recurrent hidden state of linear attention at the start of the current chunk such that we can recover to this value
            attn_state: Tuple[torch.Tensor, torch.Tensor],
                KV cache values of softmax attention operations
            conv_state: Tuple[torch.Tenspr],
                intermediate states of convolutional layers
            nats_block_types: torch.Tensor
                attention types of each nats blocks
            ffn_state: torch.Tensor
                ffn states
            g: torch.Tensor,
                decay values of linear attention
            n_new_tokens: int,
                number of new tokens at this stage
        """
        if self.n_groups_nats is None:
            k_state = attn_state[0]
            nhead_attn = k_state.shape[2]
            n_heads_nats = nats_block_types.shape[2]
            self.n_groups_nats = nhead_attn // n_heads_nats

        if self.op_for_incomplete_chunk == 'attn' and recurrent_state_block_start is not None:
            recurrent_state = recurrent_state_block_start.clone()

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
            if max_block_size > 0:
                # this only applies if we have
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
                self.n_attn_blocks_max = self.n_attn_blocks.max()
            else:
                if self.n_attn_blocks is not None:
                    incomplete_block_shift_idx = self.n_attn_blocks.max()
                    new_kv_start_idx_block = self.n_attn_blocks.min()
                    incomplete_block_shift_idx_ = incomplete_block_shift_idx - new_kv_start_idx_block
                    new_kv_end_idx_block = nats_block_types.shape[1] + self.n_attn_blocks_max
                else:
                    new_kv_start_idx_block = 0
                    new_kv_end_idx_block = nats_block_types.shape[1]
                    incomplete_block_shift_idx = 0
                    incomplete_block_shift_idx_ = 0
                if has_incomplete_blocks:
                    for kv_cache in (self.attn_k, self.attn_v):
                        kv_cache = rearrange(kv_cache, 'b (t1 t2) (h1 h2) d -> b t1 t2 h1 h2 d',
                                             t2=self.nats_block_size, h2=self.n_groups_nats
                                             )
                        new_data = kv_cache[:, new_kv_start_idx_block: new_kv_end_idx_block]
                        fill_value = new_data[:, -1, :].clone()
                        kv_cache[:, incomplete_block_shift_idx_, :] = fill_value
                        kv_cache = rearrange(kv_cache, 'b t1 t2 h1 h2 d -> b (t1 t2) (h1 h2) d',
                                             t2=self.nats_block_size, h2=self.n_groups_nats
                                             )

                if self.n_attn_blocks is None:
                    k_state = attn_state[0]
                    n_heads_nats = nats_block_types.shape[2]

                    self.n_attn_blocks = torch.zeros([k_state.shape[0], n_heads_nats], device=k_state.device, dtype=torch.int64)
                n_attn_blocks_max = self.n_attn_blocks.max()
                if has_incomplete_blocks:
                    self.valid_attn_tokens[:, n_attn_blocks_max, ] = 1
                    self.valid_attn_tokens[:, n_attn_blocks_max + 1:, ] = 0
                else:
                    self.valid_attn_tokens[:, n_attn_blocks_max:, ] = 0


            self._n_tokens_in_nats_block = (self._n_tokens_in_nats_block + n_new_tokens) % self.nats_block_size
            self.n_attn_tokens_max = self.n_attn_blocks_max * self.nats_block_size + self._n_tokens_in_nats_block
            if g is not None:
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
            if g is not None:
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
    A cache used for storing hidden states produced by flash linear attention models and KV cache values for softmax attention models
    """

    is_compileable = True

    def __init__(
            self,
            seen_tokens: int = 0,
            **kwargs) -> 'NAtSCache':
        parent_init = super().__init__
        sig = inspect.signature(parent_init)
        param_names = list(sig.parameters.keys())

        if 'layer_class_to_replicate' in param_names:
            self.use_layer_class_to_replicate = True
            super().__init__(layer_class_to_replicate=NAtSLayerCache, **kwargs)
        elif 'layer_classes' in param_names:
            self.use_layer_class_to_replicate = False
            super().__init__(layer_classes=NAtSLayerCache, **kwargs)
        else:
            raise TypeError(
                "NAtSLayer cache initialization failed: HFCacheBase.__init__ accepts neither "
                "'layer_class_to_replicate' nor 'layer_classes'. This might be caused by an incompatible "
                "transformers version. Please check your transformers>=4.36.0"
            )

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
        update the hidden state of NAtS cache. This function is implemented to be compatible with the corresponding FLA layers.
        However, to update the token-wise hybrid NAts layer, please directly call the function 'add_natsl_layer_cache' to add new layers
        and directly read the layer caches from `self.states` to updates its internal caches. 

        Args:
            recurrent_state: torch.Tensor:
                The new recurrent state at the current time step.
            recurrent_state_block_start: torch.Tensor
                Start of the recurrent states that we can recover to.
            attn_state: Tuple[torch.Tensor],
                The new attention key/value states to cache.
            conv_state: Tuple[torch.Tensor],
                The new convolution state to cache.
            ffn_state: Tuple[torch.Tensor],
                The new feed-forward state to cache.
            layer_idx: int,
                The index of the layer to cache the states for.
            offset: int,
                The number of new tokens being processed.
            cache_kwargs: Dict[str, Any],
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
            self.add_natsl_layer_cache(
                seen_tokens=self._seen_tokens,
                n_tokens_in_nats_block=0,
                nats_block_size=self.nats_block_size
            )
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

    assert (n_tokens_in_nats_block1 == n_tokens_in_nats_block2)

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
    nats_block_type12[:, 29] = 0
    k12_o, v12_o = cache1.update_attn_cache((k12, v12), nats_block_type12)

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
    k21_o, v21_o = cache2.update_attn_cache((k21, v21), nats_block_type21)
    cache2.update(attn_state=(k21, v21), nats_block_types=nats_block_type21, n_new_tokens=T1)

    for i in range(T2):
        k22_o, v2_o = cache2.update_attn_cache((k22[:, [i]], v22[:, [i]]), nats_block_type12[:, [(T1 + i) // nats_block_size]])
        cache2.update(attn_state=(k22[:, [i]], v22[:, [i]]),
                      nats_block_types=nats_block_type12[:, [(T1 + i) // nats_block_size]], n_new_tokens=1)
    cache_k2 = cache1.attn_k
    valid_attn_tokens1 = cache1.valid_attn_tokens
    valid_attn_tokens2 = cache2.valid_attn_tokens

    n_attn_blocks1 = cache1.n_attn_blocks
    n_attn_blocks2 = cache2.n_attn_blocks

    n_tokens_in_nats_block1 = cache1._n_tokens_in_nats_block
    n_tokens_in_nats_block2 = cache2._n_tokens_in_nats_block


    assert (cache_k1 - cache_k2).abs().sum() == 0
    if valid_attn_tokens1 is not None:
        if valid_attn_tokens2.shape[1] >= valid_attn_tokens1.shape[1]:
            assert (valid_attn_tokens2[:, :valid_attn_tokens1.shape[1]] == valid_attn_tokens1).all()
            if  valid_attn_tokens2.shape[1] > valid_attn_tokens1.shape[1]:
                assert valid_attn_tokens2[:, valid_attn_tokens1.shape[1]:].int().sum() == 0
        else:
            (valid_attn_tokens2[:, ] == valid_attn_tokens1[:, :valid_attn_tokens2.shape[1]]).all()
            assert valid_attn_tokens1[:, valid_attn_tokens2.shape[1]:].int().sum() == 0

    assert (n_attn_blocks1 == n_attn_blocks2).all()

    assert (n_tokens_in_nats_block1 == n_tokens_in_nats_block2)


if __name__ == "__main__":
    # only works on post-Ampere GPUs right now
    # test_compute_h()
    # test_nats_cache_attn()
    #test_nats_cache_attn_chunk_wise()
    test_nats_cache_attn_iter()
