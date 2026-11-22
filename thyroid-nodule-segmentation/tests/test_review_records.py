import tempfile
import unittest
from pathlib import Path

from review_records import prepare_sheet,save_key


class ReviewRecords(unittest.TestCase):
    def test_repeat_preserves_answer_and_assignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp); sheet=folder/'response-sheet.csv'
            original=[dict(case=1,i=2,A='gray')]
            save_key(folder,original)
            prepare_sheet(sheet,original,False)
            completed=sheet.read_bytes().replace(b'1,,,',b'1,A,missed tissue,apex')
            sheet.write_bytes(completed)
            save_key(folder,original)
            prepare_sheet(sheet,original,False)
            with self.assertRaises(ValueError):
                save_key(folder,[dict(case=1,i=3,A='contrast')])
            self.assertEqual(sheet.read_bytes(),completed)
