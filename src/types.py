from typing import NewType
from dataclasses import dataclass

RID = NewType('RID', int)
GID = NewType('GID', int)
SGID = NewType('SGID', str)
VOL = NewType('VOL', int)

# @dataclass(frozen=True, order=True)
# class ImgID:
#     gid:GID
#     sgid:SGID
#     vol:VOL
#     filename:str
