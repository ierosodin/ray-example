import ray
import numpy as np
from PIL import Image
import time
import os
import torch


class DataLoader():
    def __init__(self):
        self.preprocess_section = {
            "4": {
                "input_list": {
                    "source": ["start", "image"],
                },
                "output_list": ["image"],
                "task_fn": "resize",
                "param": {
                },
            },
            "5": {
                "input_list": {
                    "source": ["4", "image"],
                },
                "output_list": ["image"],
                "task_fn": "transpose",
                "param": {
                },
            },
            "6": {
                "input_list": {
                    "source": ["4", "image"],
                },
                "output_list": ["image"],
                "task_fn": "transpose",
                "param": {
                },
            },
            "7": {
                "input_list": {
                    "source1": ["5", "image"],
                    "source2": ["6", "image"],
                    "source3": ["4", "image"],
                },
                "output_list": ["image", "rimage", "depth"],
                "task_fn": "stop",
                "param": {
                },
            },
        }

    @ray.remote(num_cpus=1)
    def start():
        rs = np.random.RandomState(os.getpid() + int.from_bytes(os.urandom(4), byteorder='little') >> 1)
        image = np.ones((640, 480, 3)) * rs.randint(1, 100)
        return image

    @ray.remote(num_cpus=1)
    def resize(source):
        img = Image.fromarray(np.uint8(source))
        img = img.resize((4, 4))
        return img

    @ray.remote(num_cpus=1)
    def transpose(source):
        img = Image.fromarray(np.uint8(source))
        img = img.transpose(2)
        return img

    @ray.remote(num_cpus=1)
    def stop(source1, source2, source3):
        img1 = Image.fromarray(np.uint8(source1))
        img2 = Image.fromarray(np.uint8(source2))
        img3 = Image.fromarray(np.uint8(source3))
        return torch.as_tensor(np.array(img1)), {'1': np.array(img2)}, np.array(img3)

    def set_graph(self):
        self.output_cache = {}
        ret = self.start.options(num_return_vals=1).remote()
        self.output_cache["{}_{}".format(4, self.preprocess_section["4"]["output_list"][0])] = ret
        ret = self.resize.options(num_return_vals=1).remote(*[self.output_cache["{}_{}".format(4, "image")]])
        self.output_cache["{}_{}".format(5, self.preprocess_section["5"]["output_list"][0])] = ret
        ret = self.transpose.options(num_return_vals=1).remote(*[self.output_cache["{}_{}".format(5, "image")]])
        self.output_cache["{}_{}".format(6, self.preprocess_section["6"]["output_list"][0])] = ret
        ret = self.stop.options(num_return_vals=1).remote(*[self.output_cache["{}_{}".format(5, "image")],
                                                            self.output_cache["{}_{}".format(6, "image")],
                                                            self.output_cache["{}_{}".format(4, "image")]])
        self.output_cache["stop"] = ret

    def run(self):
        self.set_graph()
        return ray.get(self.output_cache["stop"])



if __name__ == '__main__':
    ray.init(num_cpus=4)
    data_loader = DataLoader()
    ret = data_loader.run()
    ray.shutdown()
