from typing import Any, Dict, List, Optional
from langchain_community.llms.vllm import VLLM
from langchain_core.utils import pre_init

class MyVLLM(VLLM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.gpu_memory_utilization = kwargs.pop("gpu_memory_utilization", 0.9)
        

    @pre_init
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that python package exists in environment."""

        try:
            from vllm import LLM as VLLModel
        except ImportError:
            raise ImportError(
                "Could not import vllm python package. "
                "Please install it with `pip install vllm`."
            )

        values["client"] = VLLModel(
            model=values["model"],
            tensor_parallel_size=values["tensor_parallel_size"],
            trust_remote_code=values["trust_remote_code"],
            dtype=values["dtype"],
            download_dir=values["download_dir"],
            # gpu_memory_utilization=self.gpu_memory_utilization,
            **values["vllm_kwargs"],
        )