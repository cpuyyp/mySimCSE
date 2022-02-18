import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import transformers
from transformers import RobertaTokenizer
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaLMHead
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertLMPredictionHead
from transformers.activations import gelu
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
# from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions
from transformers.file_utils import ModelOutput
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple

@dataclass
class MyBaseModelOutputWithPoolingAndCrossAttentions(BaseModelOutputWithPoolingAndCrossAttentions):
     style_emb: torch.FloatTensor = None
     content_emb: torch.FloatTensor = None

# Joey: redefine the output to suit my needs
@dataclass
class SequenceClassifierOutput(ModelOutput):
    """
    Base class for outputs of sentence classification models.
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    style_adversarial_loss: Optional[torch.FloatTensor] = None
    content_adversarial_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x

class DisentangleLayer(nn.Module):
    """
    Joey: Head after pooler for divide style and content informatino into 
    two seperate embeddings.
    """

    def __init__(self, config, model_args):
        super().__init__()
        # consider more ways to disentangle.
        # if config.disentangle_type ==  '':
        self.dense2style = nn.Linear(config.hidden_size, model_args.style_size)
        self.activation_style = nn.Tanh()
        self.dense2content = nn.Linear(config.hidden_size, model_args.content_size)
        self.activation_content = nn.Tanh()

    def forward(self, features, **kwargs):
        style_emb = self.dense2style(features)
        style_emb = self.activation_style(style_emb)
        content_emb = self.dense2content(features)
        content_emb = self.activation_style(content_emb)
        return (style_emb, content_emb)

class StyleClassifier(nn.Module):
    """
    Joey: use style embedding to predict POS frequency vector
    """
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.style_size, config.POS_vocab_size)
        self.activation = nn.ReLU()
    def forward(self, style_emb, **kwargs):
        POS_vec = self.dense(style_emb)
        POS_vec = self.activation(POS_vec)
        return POS_vec

class ContentClassifier(nn.Module):
    """
    Joey: use content embedding to predict BOW frequency vector
    """
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.content_size, config.BOW_vocab_size)
        self.activation = nn.ReLU()
    def forward(self, content_emb, **kwargs):
        BOW_vec = self.dense(content_emb)
        BOW_vec = self.activation(BOW_vec)
        return BOW_vec

class StyleAdversary(nn.Module):
    """
    Joey: use content embedding to predict POS frequency vector to provide adversarial loss
    Detach to prevent gradients flow back to bert
    """
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.content_size, config.POS_vocab_size)
        self.activation = nn.ReLU()
    def forward(self, content_emb, **kwargs):
        POS_vec = self.dense(content_emb.detach())
        POS_vec = self.activation(POS_vec)
        return POS_vec
    
class ContentAdversary(nn.Module):
    """
    Joey: use style embedding to predict BOW frequency vector to provide adversarial loss
    Detach to prevent gradients flow back to bert
    """
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.style_size, config.BOW_vocab_size)
        self.activation = nn.ReLU()
    def forward(self, style_emb, **kwargs):
        BOW_vec = self.dense(style_emb.detach())
        BOW_vec = self.activation(BOW_vec)
        return BOW_vec

        
class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError


def cl_init(cls, config):
    """
    Contrastive learning class init function.
    """
    cls.pooler_type = cls.model_args.pooler_type
    cls.pooler = Pooler(cls.model_args.pooler_type)
    cls.label_smoothing = cls.model_args.label_smoothing
    cls.BOW_vocab_size = cls.model_args.BOW_vocab_size
    cls.POS_vocab_size = cls.model_args.POS_vocab_size
    cls.epsilon = cls.model_args.epsilon
    if cls.model_args.pooler_type == "cls":
        cls.mlp = MLPLayer(config)
    cls.sim = Similarity(temp=cls.model_args.temp)
    cls.disentangleLayer = DisentangleLayer(config, cls.model_args)
    cls.styleClassifier = StyleClassifier(cls.model_args)
    cls.contentClassifier = ContentClassifier(cls.model_args)
    cls.styleAdversary = StyleAdversary(cls.model_args)
    cls.contentAdversary = ContentAdversary(cls.model_args)
    cls.annealed_weight = 1
    cls.init_weights()
    # return cls

def get_entropy_loss(preds, epsilon):
    """
    Returns the entropy loss: negative of the entropy present in the
    input distribution
    Note: this is already the negative
    """
    return torch.mean(torch.sum(preds * torch.log(preds + epsilon), dim=1))

# Joey: the training forward
def cl_forward(cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    mlm_input_ids=None,
    mlm_labels=None,
    pos_labels = None,
    bow_labels = None,
):
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    ori_input_ids = input_ids
    batch_size = input_ids.size(0)
    # Number of sentences in one instance
    # 2: pair instance; 3: pair instance with a hard negative
    num_sent = input_ids.size(1)

    mlm_outputs = None

    # Joey: construct BOW feature from input_ids
    bow_labels = torch.zeros(batch_size,1, cls.BOW_vocab_size)
    for irow in range(batch_size):
        for jcol in input_ids[irow,0]:
            bow_labels[irow,0,jcol] += 1
    bow_labels = torch.cat((bow_labels,bow_labels), 1)

    # convert to probability for CrossEntropyLoss
    bow_labels = bow_labels.softmax(dim=(2))
    pos_labels = pos_labels.softmax(dim=(2))

    # Flatten input for encoding
    input_ids = input_ids.view((-1, input_ids.size(-1))) # (bs * num_sent, len)
    attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs * num_sent, len)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1))) # (bs * num_sent, len)

    # Get raw embeddings
    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )

    # MLM auxiliary objective
    if mlm_input_ids is not None:
        mlm_input_ids = mlm_input_ids.view((-1, mlm_input_ids.size(-1)))
        mlm_outputs = encoder(
            mlm_input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
            return_dict=True,
        )

    # Pooling
    pooler_output = cls.pooler(attention_mask, outputs)
    pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1))) # (bs, num_sent, hidden)

    # If using "cls", we add an extra MLP layer
    # (same as BERT's original implementation) over the representation.
    # Joey: Why don't put cls pooler insider the pooler class??
    # Joey: Because model_args.mlp_only_train. 
    # During evaluation, using only the CLS as sentence embedding may be better?
    if cls.pooler_type == "cls":
        pooler_output = cls.mlp(pooler_output)

    # construct bow_feature from input_ids

    # Joey: Separate style and content
    # Start calculate some loss
    # why pos loads 2xbatch size ????
    
    # smoothed_bow_labels = bow_labels * (1-cls.label_smoothing) + cls.label_smoothing/cls.BOW_vocab_size
    # smoothed_pos_labels = pos_labels * (1-cls.label_smoothing) + cls.label_smoothing/cls.POS_vocab_size
    smoothed_bow_labels = bow_labels 
    smoothed_pos_labels = pos_labels 
    
    style_emb, content_emb = cls.disentangleLayer(pooler_output)
    pos_pred = cls.styleClassifier(style_emb)
    bow_pred = cls.contentClassifier(content_emb)

    pos_pred_adv = cls.styleAdversary(content_emb)
    bow_pred_adv = cls.contentAdversary(style_emb)

    softmax_loss_fct = nn.Softmax(dim=1)
    style_entropy_loss = get_entropy_loss(softmax_loss_fct(pos_pred_adv), cls.epsilon)
    content_entropy_loss = get_entropy_loss(softmax_loss_fct(bow_pred_adv), cls.epsilon)


    crossentropy_loss_fct = nn.CrossEntropyLoss()
    smoothed_pos_labels = smoothed_pos_labels.view(pos_pred.shape)
    # print(smoothed_pos_labels.shape)
    # print(pos_pred.shape)
    # smoothed_pos_labels = smoothed_pos_labels[:,0,:]
    smoothed_pos_labels = smoothed_pos_labels.to(cls.device) 
    smoothed_bow_labels = smoothed_bow_labels.to(cls.device) 

    # mse_loss_fct = nn.MSELoss()

    style_classifier_loss = crossentropy_loss_fct(pos_pred, smoothed_pos_labels)
    content_classifier_loss = crossentropy_loss_fct(bow_pred, smoothed_bow_labels)

    style_adversarial_loss = crossentropy_loss_fct(pos_pred_adv, smoothed_pos_labels)
    content_adversarial_loss = crossentropy_loss_fct(bow_pred_adv, smoothed_bow_labels)


    # Separate sentence representation
    z1, z2 = pooler_output[:,0], pooler_output[:,1]

    # Hard negative
    if num_sent == 3:
        z3 = pooler_output[:, 2]

    # Gather all embeddings if using distributed training
    if dist.is_initialized() and cls.training:
        # Gather hard negative
        if num_sent >= 3:
            z3_list = [torch.zeros_like(z3) for _ in range(dist.get_world_size())]
            dist.all_gather(tensor_list=z3_list, tensor=z3.contiguous())
            z3_list[dist.get_rank()] = z3
            z3 = torch.cat(z3_list, 0)

        # Dummy vectors for allgather
        z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
        z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
        # Allgather
        dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
        dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())

        # Since allgather results do not have gradients, we replace the
        # current process's corresponding embeddings with original tensors
        z1_list[dist.get_rank()] = z1
        z2_list[dist.get_rank()] = z2
        # Get full batch embeddings: (bs x N, hidden)
        z1 = torch.cat(z1_list, 0)
        z2 = torch.cat(z2_list, 0)

    cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0)) # (bs * bs)
    # Hard negative
    if num_sent >= 3:
        z1_z3_cos = cls.sim(z1.unsqueeze(1), z3.unsqueeze(0))
        cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)

    labels = torch.arange(cos_sim.size(0)).long().to(cls.device) # [0,1,2, ... bs-1]
    # loss_fct = nn.CrossEntropyLoss()

    # Calculate loss with hard negatives
    if num_sent == 3:
        # Note that weights are actually logits of weights
        z3_weight = cls.model_args.hard_negative_weight
        weights = torch.tensor(
            [[0.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1)) + [0.0] * i + [z3_weight] + [0.0] * (z1_z3_cos.size(-1) - i - 1) for i in range(z1_z3_cos.size(-1))]
        ).to(cls.device)
        cos_sim = cos_sim + weights

    # Joey: Why loss(cos_sim, labels)?
    # Joey: labels is array([0,1,2,...,bs-1]). cos_sim has shape (bs * bs). 
    # In the two columns case, take the ith sentence in the first column, 
    # we use crossentropy to classify which sentence in the second column match it.
    # And it's the ith. Because both of them are the same sentence but different runs.
    loss = crossentropy_loss_fct(cos_sim, labels) \
                + 1*cls.annealed_weight * style_classifier_loss \
                + 1*cls.annealed_weight * content_classifier_loss \
                + 1*cls.annealed_weight * style_entropy_loss \
                + 1*cls.annealed_weight * content_entropy_loss

    # Calculate loss for MLM
    if mlm_outputs is not None and mlm_labels is not None:
        mlm_labels = mlm_labels.view(-1, mlm_labels.size(-1))
        prediction_scores = cls.lm_head(mlm_outputs.last_hidden_state)
        masked_lm_loss = crossentropy_loss_fct(prediction_scores.view(-1, cls.config.vocab_size), mlm_labels.view(-1))
        loss = loss + cls.model_args.mlm_weight * masked_lm_loss

    if not return_dict:
        output = (cos_sim,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output
    return SequenceClassifierOutput(
        loss=loss,
        style_adversarial_loss = style_adversarial_loss,
        content_adversarial_loss = content_adversarial_loss,
        logits=cos_sim,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

# Joey: the evaluating forward
def sentemb_forward(
    cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):

    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )

    pooler_output = cls.pooler(attention_mask, outputs)
    

    if cls.pooler_type == "cls" and not cls.model_args.mlp_only_train:
        pooler_output = cls.mlp(pooler_output)

    style_emb, content_emb = cls.disentangleLayer(pooler_output)

    if not return_dict:
        return (outputs[0], pooler_output) + outputs[2:]

    return MyBaseModelOutputWithPoolingAndCrossAttentions(
        pooler_output=pooler_output,
        style_emb=style_emb,
        content_emb=content_emb,
        last_hidden_state=outputs.last_hidden_state,
        hidden_states=outputs.hidden_states,
    )


class BertForCL(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.bert = BertModel(config, add_pooling_layer=False)

        if self.model_args.do_mlm:
            self.lm_head = BertLMPredictionHead(config)

        cl_init(self, config)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
        pos_labels = None,
        bow_labels = None,
    ):
        if sent_emb:
            return sentemb_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
                pos_labels = pos_labels,
                bow_labels = bow_labels,
            )



class RobertaForCL(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.roberta = RobertaModel(config, add_pooling_layer=False)

        if self.model_args.do_mlm:
            self.lm_head = RobertaLMHead(config)

        cl_init(self, config)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
        pos_labels = None,
        bow_labels = None,
    ):
        if sent_emb:
            return sentemb_forward(self, self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self, self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
                pos_labels = pos_labels,
                bow_labels = bow_labels,
            )
