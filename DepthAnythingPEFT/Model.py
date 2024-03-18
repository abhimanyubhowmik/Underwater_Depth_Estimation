from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from peft import get_peft_model

class DepthAnythingPEFT:

    def __init__(self, model_checkpoint) -> None:
        self.model_checkpoint = model_checkpoint
        self.image_processor = AutoImageProcessor.from_pretrained(model_checkpoint)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_checkpoint,
        ignore_mismatched_sizes=True,)  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint


    def trainable_parameters(self, model):
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
        )

    def trainable_modules(self,model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.shape)


    def peft_model(self, peft_config):
        return get_peft_model(self.model, peft_config)


