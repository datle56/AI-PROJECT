import torch
from torch import nn
from transformers import Wav2Vec2ForCTC, Wav2Vec2Config
from transformers.modeling_outputs import CausalLMOutput
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Encoder
from transformers import Wav2Vec2Processor
# from model import CustomWav2Vec2ForCTC  # Import mô hình từ file model.py



# checkpoint_dir = "/content/dataset-folder/checkpoint-50000"
# processor = Wav2Vec2Processor.from_pretrained('/content/dataset-folder')


class CustomWav2Vec2ForCTC(Wav2Vec2ForCTC):
    def __init__(self, config):
        super().__init__(config)
        # Freeze the feature extractor
        self.wav2vec2.feature_extractor._freeze_parameters()

        # Add the first Transformer encoder layer
        encoder_layer_1 = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=8,  # Number of attention heads
            dim_feedforward=2048,  # Hidden layer size in feedforward network
            dropout=0.1,  # Dropout probability
            activation='relu'  # Activation function
        )
        # Create a stack of the first Transformer encoder layers
        self.transformer_1 = nn.TransformerEncoder(
            encoder_layer_1, num_layers=2  # Number of Transformer layers
        )

        # Add the second Transformer encoder layer
        encoder_layer_2 = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            activation='relu'
        )
        # Create a stack of the second Transformer encoder layers
        self.transformer_2 = nn.TransformerEncoder(
            encoder_layer_2, num_layers=2
        )

        # Update the classifier to match the Transformer output size
        self.classifier = nn.Linear(config.hidden_size, config.vocab_size)

        # Initialize ctc_loss
        self.ctc_loss = nn.CTCLoss(
            blank=self.config.pad_token_id,
            reduction=self.config.ctc_loss_reduction,
            zero_infinity=self.config.ctc_zero_infinity
        )

    def forward(
        self,
        input_values,
        attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # Get outputs from Wav2Vec2 model
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]  # (batch_size, sequence_length, hidden_size)

        # Pass through the first Transformer encoder
        transformer_output_1 = self.transformer_1(hidden_states.permute(1, 0, 2))  # (sequence_length, batch_size, hidden_size)
        transformer_output_1 = transformer_output_1.permute(1, 0, 2)  # Back to (batch_size, sequence_length, hidden_size)

        # Pass through the second Transformer encoder
        transformer_output_2 = self.transformer_2(transformer_output_1.permute(1, 0, 2))  # (sequence_length, batch_size, hidden_size)
        transformer_output_2 = transformer_output_2.permute(1, 0, 2)  # Back to (batch_size, sequence_length, hidden_size)

        # Compute logits
        logits = self.classifier(transformer_output_2)  # (batch_size, sequence_length, vocab_size)

        loss = None
        if labels is not None:
            # Compute CTC loss
            log_probs = nn.functional.log_softmax(logits, dim=-1).transpose(0, 1)
            input_lengths = torch.full(
                size=(logits.size(0),), fill_value=logits.size(1), dtype=torch.long
            )
            target_lengths = torch.sum(labels != -100, dim=-1)
            loss = self.ctc_loss(log_probs, labels, input_lengths, target_lengths)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )