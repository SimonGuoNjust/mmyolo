from test_datasets.test_transforms.test_transforms import TestYOLOv5RandomAffineMask

a = TestYOLOv5RandomAffineMask()
a.setUp()
a.test_transform_with_boxlist_mask()