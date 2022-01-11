from dataset import generate_heat_map as ghm
import os
import pytest


class TestGenerateHeatmap:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.prepare_params, self.global_params = ghm.load_yaml_prepare_params()
        self.dataset_path = os.path.join(self.global_params["DATA_PATH"], 'raw')
        self.trainset_path = os.path.join(self.dataset_path, 'train')
        self.testset_path = os.path.join(self.dataset_path, 'test')
        self.annotations_path = os.path.join(self.dataset_path, 'annotations')

    @pytest.mark.prepare
    def test_param_load(self):
        assert issubclass(type(self.prepare_params), dict)
        assert issubclass(type(self.global_params), dict)

    @pytest.mark.prepare
    def test_dataframe_load_train(self):
        ann = next(filter(lambda x: '_test' not in x, os.listdir(self.annotations_path)))
        df = ghm.dataframe_load_train(os.path.join(self.annotations_path, ann))
        assert df['x'].notnull().all()
        assert df['y'].notnull().all()
        assert df['frame'].notnull().all()

    @pytest.mark.prepare
    def test_dataframe_load_test(self):
        ann = next(filter(lambda x: '_test' in x, os.listdir(self.annotations_path)))
        df = ghm.dataframe_load_test(os.path.join(self.annotations_path, ann))
        assert df['x'].notnull().all()
        assert df['y'].notnull().all()
        assert df['frame'].notnull().all()

    @pytest.mark.prepare
    def test_generate_heat_map(self):
        size = (960, 540)
        gamma = 3
        ann = next(filter(lambda x: '_test' not in x, os.listdir(self.annotations_path)))

        df = ghm.dataframe_load_train(os.path.join(self.annotations_path, ann))
        heatmaps = ghm.generate_heatmap(df, size, size, gamma)
        assert set(heatmaps.keys()) == set(df['frame'])  # Check it generated all the heatmaps
        for key in heatmaps.keys():
            (heatmaps[key] > 0).all()  # Heatmaps values are always greater than 0

    @pytest.mark.prepare
    @pytest.mark.slow
    def test_main(self):
        # For main it is sufficient to check that ends without errors
        ghm.main()
