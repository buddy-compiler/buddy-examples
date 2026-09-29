def register():
    from vllm.model_executor.models import ModelRegistry

    ModelRegistry.register_model("examples.models.goban.bert-base-uncased-emotion.serve",
                                 "examples.models.goban.bert-base-uncased-emotion.serve:Model")
