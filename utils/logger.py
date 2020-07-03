import time
from contextlib import contextmanager


class logger:
    def __init__(self):
        self.runtime_data = dict()
        self.attribute_data = dict()

    def log_attribute(self, attribute_name, value):
        if attribute_name in self.attribute_data.keys():
            self.attribute_data[attribute_name].append(value)
        else:
            print("Logging " + attribute_name)
            self.attribute_data[attribute_name] = [value]

    @contextmanager
    def log_runtime(self, step_name):
        start = time.perf_counter()
        yield
        stop = time.perf_counter()
        elapsed_time = (stop - start) * 1000.0  # ms
        if step_name in self.runtime_data.keys():
            self.runtime_data[step_name].append(elapsed_time)
        else:
            print("Logging " + step_name)
            self.runtime_data[step_name] = [elapsed_time]

    def print_last(self):
        p_string = ""
        for key in self.runtime_data.keys():
            p_string += key + ": " + str(self.runtime_data[key][-1])
            p_string += ", "
        for key in self.attribute_data.keys():
            p_string +=key + ": " + str(self.attribute_data[key][-1])
            p_string += ", "
        p_string = p_string[: -2] + "\n"

        print(p_string)


    def export(self, export_path):
        with open(export_path, 'w+') as f:

            keys_rt = list(self.runtime_data.keys())
            keys_attr = list(self.attribute_data.keys())

            iterations = len(self.runtime_data[keys_rt[0]])

            # write keys as headers
            f.write(", ".join(keys_rt + keys_attr) + "\n")

            # write the values
            for idx in range(iterations):
                p_string = ""
                for key in keys_rt:
                    p_string += str(self.runtime_data[key][idx])
                    p_string += ", "
                for key in keys_attr:
                    p_string += str(self.attribute_data[key][idx])
                    p_string += ", "
                p_string = p_string[: -2] + "\n"

                f.write(p_string)
