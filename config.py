import datetime
from pathlib import Path
import time
#from .utils import get_period
# Getting path of this python file
FILE = Path(__file__)

'''
Some path configuration code that generates the correct save paths and is system agnostic.
this keeps you from having to explicitly write out data save paths. They get generated correctly
no matter where the ROOT folder is on a system. everything is define relative to ROOT.

I do this because when I put this code on an HPC, I need all the path information to be consistent
this code generates it automatically. 

All you have to do is import PathConfigs defined below and save output to self.DATA

'''

class DefaultPathConfiguration:  
    # This is the set of default file paths that I use. ROOT is defined as the parent of this config file.
    def __init__(self, FILE) -> None:
        # defining relative path to root, FILE is the path to this python file.
        self.ROOT = FILE.parents[0]
        self.DATA = self.ROOT / "Data"
        self.TESTS = self.ROOT / "Tests"

    @classmethod
    def get_local_config(cls, local_dir):
        # returns an instance of modified paths based on local_dir
        ret = cls(FILE)
        ret.CACHE = ret.CACHE/local_dir
        ret.OUTPUT = ret.OUTPUT/local_dir


class LocalPathConfiguration(DefaultPathConfiguration):
    def __init__(self, local_cache_dir):
        super().__init__(FILE)
        self.local_cache = self.CACHE / local_cache_dir


PathConfigs = DefaultPathConfiguration(FILE)