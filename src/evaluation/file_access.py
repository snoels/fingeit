""" Module for file access operations
"""

import os
import uuid


class FileAccess:
    RECOVERY_MARK = "<ADDING AFTER RECOVERY>"

    def __init__(self, file:str|None=None) -> None:
        self.file = file if file else f"{uuid.uuid4()}.txt"


    def store(self, lines:list[str]) -> None:
        with open(self.file, 'a') as f:
            f.write('\n'.join(lines))
            f.write('\n')

    def read(self) -> list[str]:
        with open(self.file, 'r') as f:
            lines =  f.readlines()
            return [line.rstrip('\n') for line in lines if self.RECOVERY_MARK not in line]
    
    def remove(self) -> None:
        try:
            os.remove(self.file)
        except Exception as e:
            print(f"Could not remove temp file {self.file}", e)

    def mark_recovery(self):
         with open(self.file, 'a') as f:
            f.write(self.RECOVERY_MARK)
            f.write("\n")