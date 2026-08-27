import unittest

from engine.solver.det_solver import _should_run_epoch


class TrainingScheduleTest(unittest.TestCase):
    def test_frequency_extra_and_final_epochs(self):
        should_run = lambda epoch: _should_run_epoch(epoch, 132, 5, [119, 120])
        self.assertTrue(should_run(4))
        self.assertTrue(should_run(119))
        self.assertTrue(should_run(120))
        self.assertTrue(should_run(131))
        self.assertFalse(should_run(118))


if __name__ == '__main__':
    unittest.main()
