from typing import Type, Any, Dict

class ModelFactory():
    models = {}

    @staticmethod
    def register(model_class: Type):
        if model_class.name in ModelFactory.models:
            raise Exception(f"{model_class.name} is already registered")
        
        ModelFactory.models[model_class.name] = model_class
    
    @staticmethod
    def get_model(model_name: str, **kwargs):
        if model_name not in ModelFactory.models.keys():
            raise Exception(f"Unknown model {model_name}")
        
        model_class = ModelFactory.models[model_name]
        return model_class(**kwargs)
    
    @staticmethod
    def is_legal(model_name: str, base_params: Dict[str, Any], share_params: Dict[str, Any], *args, **kwargs):
        if model_name not in ModelFactory.models.keys():
            raise Exception(f"Unknown model {model_name}")
        
        model_class = ModelFactory.models[model_name]

        return model_class.is_legal(base_params, share_params)
