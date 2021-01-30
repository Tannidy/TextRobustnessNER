from TextRobustness.dataset import Dataset


# TODO
class LocalDataset(Dataset):
    """Loads a dataset from datasets of CSV format and prepares it as a Dataset.

   - csv_path: input file path.
   - label_map: Mapping if output labels should be re-mapped. Useful
      if model was trained with a different label arrangement than
      provided in the ``datasets`` version of the dataset.
   - contains_header: CSV file contains headers or not, if not
      dataset_columns should be provided.
   - dataset_columns: CSV columns name.
    """

    def __init__(
            self,
            csv_path,
            task=None,
            label_map=None,
            contains_header=False,
            dataset_columns=None,
    ):
        pass
