import unittest

import numpy as np

from train_minimal_cnn import resolve_test_authors, split_by_author


class LeaveOneAuthorOutTests(unittest.TestCase):
    def setUp(self):
        self.authors = np.repeat(
            np.array(["melissa", "ivett", "benja", "martin"]), 4
        )
        self.labels = np.tile(
            np.array(["abiertos", "cerrados", "abiertos", "cerrados"]), 4
        )

    def test_leave_one_out_selects_each_author_once(self):
        test_authors = resolve_test_authors(
            self.authors, test_author=None, leave_one_author_out=True
        )

        self.assertEqual(test_authors, ["benja", "ivett", "martin", "melissa"])

    def test_each_fold_keeps_test_author_out_of_train_and_validation(self):
        test_authors = resolve_test_authors(
            self.authors, test_author=None, leave_one_author_out=True
        )

        for test_author in test_authors:
            with self.subTest(test_author=test_author):
                train, validation, test = split_by_author(
                    self.labels,
                    self.authors,
                    test_author,
                    validation_fraction=0.2,
                    seed=42,
                )
                self.assertEqual(set(self.authors[test]), {test_author})
                self.assertNotIn(test_author, set(self.authors[train]))
                self.assertNotIn(test_author, set(self.authors[validation]))
                self.assertEqual(set(self.labels[test]), {"abiertos", "cerrados"})

    def test_single_author_holdout_remains_available(self):
        self.assertEqual(
            resolve_test_authors(
                self.authors, test_author="martin", leave_one_author_out=False
            ),
            ["martin"],
        )

    def test_unknown_test_author_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no existe en los datos"):
            resolve_test_authors(
                self.authors, test_author="guest", leave_one_author_out=False
            )

    def test_single_author_and_leave_one_out_cannot_be_requested_together(self):
        with self.assertRaisesRegex(ValueError, "no se pueden combinar"):
            resolve_test_authors(
                self.authors, test_author="martin", leave_one_author_out=True
            )


if __name__ == "__main__":
    unittest.main()
