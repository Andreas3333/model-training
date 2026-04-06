from transformers import PreTrainedModel, BertConfig, BertModel

class BertForMultiClassSeqClassificationConfig(BertConfig):
    model_type = "bert"
    problem_type = "classification"
    finetuning_task = "multi_class_seq_classification"
    
    def __init__(self, **kwargs)-> None:
        super().__init__(**kwargs)

class BertForMultiClassSeqClassification(PreTrainedModel):
    """
    google-bert/bert-base-uncased for multi class classification
    """
    config_class = BertForMultiClassSeqClassificationConfig

    def __init__(self, config, base_bert_checkpoint) -> None:
        super().__init__(config)
        self.bert = BertModel.from_pretrained(
                config=config,
                pretrained_model_name_or_path=base_bert_checkpoint,
                add_pooling_layer=True,
                local_files_only=True,
                use_safetensors=True,
            )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        return self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )["pooler_output"]
