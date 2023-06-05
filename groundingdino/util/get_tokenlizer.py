from transformers import AutoTokenizer, BertModel

def get_tokenlizer(text_encoder_type):
    return AutoTokenizer.from_pretrained("pretrained/bert-base-uncased")


def get_pretrained_language_model(text_encoder_type):
    return BertModel.from_pretrained("pretrained/bert-base-uncased")
