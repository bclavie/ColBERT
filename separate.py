from transformers import PreTrainedModel, AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig

class MonoT5EncoderModel(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = AutoModelForSeq2SeqLM.from_pretrained(
            config._name_or_path, config=config
        ).encoder

    def forward(self, **kwargs):
        return self.encoder(**kwargs)

    def save_pretrained(self, save_directory):
        self.encoder.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop("config", None)
        if config is None:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        config._name_or_path = pretrained_model_name_or_path
        model = super().from_pretrained(
            pretrained_model_name_or_path, config=config, *model_args, **kwargs
        )
        model.encoder = AutoModelForSeq2SeqLM.from_pretrained(
            pretrained_model_name_or_path, config=config
        ).encoder
        return model

print("Loading")
encoder = MonoT5EncoderModel.from_pretrained('google/flan-t5-small')
encoder.to('cuda')
tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-xl')
print("loaded")
q = "What is the population of Tokyo?"
in_answer = "retrieve a passage that answers this question from Wikipedia"

p_1 = "The population of Japan's capital, Tokyo, dropped by about 48,600 people to just under 14 million at the start of 2022, the first decline since 1996, the metropolitan government reported Monday."
p_2 = "Tokyo, officially the Tokyo Metropolis (東京都, Tōkyō-to), is the capital and largest city of Japan."

# 1. TART-full can identify more relevant paragraph. 
features = tokenizer(['{0} [SEP] {1}'.format(in_answer, q)] * 12, padding=True, truncation=True, return_tensors="pt").to('cuda')          # print(self.model.instruction_encoder.encoder(**instruction))
print('DEBERTA ON SEPARATE TOKENIZER')
print(encoder(**features))