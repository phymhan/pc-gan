from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--how_many', type=int, default=20000, help='how many test images to run')
        # options for computing inception score
        parser.add_argument('--which_model_IS', type=str, default='inception_v3')
        parser.add_argument('--batchSize_IS', type=int, default=32)
        parser.add_argument('--pretrained_model_path_IS')
        parser.add_argument('--splits', type=int, default=10)
        parser.add_argument('--result_path', type=str, default='')
        # options for generating images
        parser.add_argument('--how_to_sample', type=str, choices=['prior', 'label'], default='prior', help='sample from what')
        parser.add_argument('--sample_label_file', type=str, default='')
        parser.add_argument('--output_dir', type=str)
        parser.set_defaults(model='test')
        # To avoid cropping, the loadSize should be the same as fineSize
        parser.set_defaults(loadSize=parser.get_default('fineSize'))
        
        self.isTrain = False
        return parser
