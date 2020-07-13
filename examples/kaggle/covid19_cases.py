import os
from urllib.request import urlretrieve

import pandas as pd

REPO_BASE = r"https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series"
URLs = dict(
    recovered=r"time_series_covid19_recovered_global.csv",
    deaths=r"time_series_covid19_deaths_global.csv",
    confirmed=r"time_series_covid19_confirmed_global.csv",
)


# ===========================================================================
# Helpers
# ===========================================================================
def download(outdir="/tmp") -> dict:
  if not os.path.isdir(outdir):
    os.makedirs(outdir)
  data = {}
  for key, url in URLs.items():
    url = os.path.join(REPO_BASE, url)
    name = os.path.basename(url)
    outpath = os.path.join(outdir, name)
    data[key] = pd.read_csv(outpath)
  return data


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
  data = download()
  for i, j in data.items():
    print(j)
