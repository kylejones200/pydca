from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd
import requests
import xlrd


def scrape_ndic(months_list: List[str], output_dir: str = "ndic_raw") -> pd.DataFrame:
    base_url = "https://www.dmr.nd.gov/oilgas/mpr/"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_data = []

    for period in months_list:
        url = f"{base_url}{period}.xlsx"
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            book = xlrd.open_workbook(file_contents=r.content)
            sheet = book.sheet_by_index(0)

            for i in range(1, sheet.nrows):
                try:
                    cell_val = sheet.cell_value(i, 0)
                    year, month, *_ = xlrd.xldate_as_tuple(cell_val, book.datemode)
                    sheet._cell_values[i][0] = datetime(year, month, 1).strftime(
                        "%Y-%m-%d"
                    )
                except Exception:
                    sheet._cell_values[i][0] = ""

            rows = [sheet.row_values(rownum) for rownum in range(sheet.nrows)]
            headers, data = rows[0], rows[1:]
            df = pd.DataFrame(data, columns=headers)
            df["SourceMonth"] = period
            all_data.append(df)

            out_file = output_path / f"{period}.csv"
            df.to_csv(out_file, index=False)

        except Exception as e:
            print(f"Error processing {period}: {e}")

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        return pd.DataFrame()
