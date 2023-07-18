import os
import json
from types import SimpleNamespace

class GatorConfig(SimpleNamespace):
    @classmethod
    def from_json(cls, config_json, extra={}):
        with open(config_json, "r") as f:
            d = json.load(f)
            d["name"] = config_json.split("/")[-1].replace(".json", "")
            d.update(extra)
            return json.loads(json.dumps(d), object_hook=lambda d: cls(**d))

    def write(self, outfile):
        with open(outfile, "w") as f:
            json.dump(self.__recursive_get_json(self.__dict__), f, indent=4)

    def get(self, key, default=None):
        return self.__dict__.get(key, default)

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def __recursive_get_json(self, d):
        json = {}
        for key, val in d.items():
            if type(val) == type(self):
                json[key] = self.__recursive_get_json(val)
            else:
                json[key] = val
        return json

    def __str__(self):
        return json.dumps(self.__recursive_get_json(self.__dict__), indent=4)

    def __getitem__(self, key):
        return self.get(key)

    def get_outfile(self, subdir=None, ext="pt", tag=None, epoch=None, quiet=True, short=False):
        outdir = f"{self.base_dir}/{self.name}"
        if subdir:
            outdir = f"{outdir}/{subdir}"
        os.makedirs(outdir, exist_ok=True)

        outfile = f"{outdir}/{self.name}"
        if not short:
            outfile += (
                f"_model{self.model.name}"
                + f"_nhidden{self.model.n_hidden_layers}"
                + f"_hiddensize{self.model.hidden_size}"
                + f"_lr{self.train.scheduler_name}{self.train.learning_rate}"
            )
        if not epoch is None:
            outfile += f"_epoch{epoch}"
        if not tag is None:
            outfile += f"_{tag}"

        outfile += f".{ext}"

        if not quiet:
            print(outfile)

        return outfile

class SimpleProgress:
    def __init__(self, iterable, n_checkpoints=10):
        self.__iterable = list(iterable)
        self.total = len(self.__iterable)
        self.n_checkpoints = n_checkpoints
        self.n_reports = 0
        self.check = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.check >= self.n_reports*self.total/self.n_checkpoints:
            self.n_reports += 1
            print(f"{100*self.check/self.total:0.1f}% complete", flush=True)

        if self.check >= self.total:
            raise StopIteration

        item = self.__iterable[self.check]
        self.check += 1

        return item

def print_title(text):
    text = f" {text} "
    print(f"{text:-^50}", flush=True)