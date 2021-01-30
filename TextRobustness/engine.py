"""
TextRobustness Engine Class
============================================

"""

from TextRobustness.adapter import Adapter
from TextRobustness.common.utils import auto_create_path
from TextRobustness.common.settings import CSV, JSON


class TextRobustnessEngine:
    """ Engine class of Text Robustness.

    Load Config, Generator, Dataset automatically.

    Attributes:
        config: TextRobustness.config.Config
        generator: TextRobustness.generator.Generator
        dataset: TextRobustness.dataset.Dataset

    """
    def __init__(self, config_obj=None):
        self.config = Adapter.get_config(config_obj)
        self.generator = Adapter.get_generator(self.config)
        self.dataset = Adapter.get_dataset(task=config_obj.task)

    def run(self, data_input):
        """ Engine start entrance, load data and apply transformations.

        Args:
            data_input: dict / list

        Returns:
            save to csv/json file or return json.
        """
        self.dataset.free()
        data_set = self.dataset.load(data_input)

        for transformed_result, trans_name in self.generator.generate(data_set):
            transformed_data = self.dataset.get_empty_dataset()

            transformed_data.load(transformed_result)

            # save to csv/json file or return json object
            if self.config.out_format in [JSON, CSV]:
                # TODO, verify out_format
                auto_create_path(self.config.out_path)
                if self.config.out_format == CSV:
                    transformed_data.save_to_csv(self.config.out_path)
                else:
                    transformed_data.save_to_json(self.config.out_path)
            else:
                return transformed_data.to_json()


if __name__ == "__main__":
    sent1_x = 'The quick brown fox jumps over the lazy dog .'
    sent1_y = "negative"
    data_sample = {'x': [sent1_x], 'y': [sent1_y]}

    engine = TextRobustnessEngine()
    print(engine.run(data_sample))
