# Modified from the original implementation: https://github.com/NVlabs/vla0
# Authors: Gokul Puthumanaillam, Navya Gupta


from typing import List, Optional

from pydantic import BaseModel


class So100Base64DataModel(BaseModel):
    base64_rgb: List[str]
    state: List[float]  # (6)
    instr: Optional[str] = None
