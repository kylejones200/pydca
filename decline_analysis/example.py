from decline_analysis import dca

series = df[df["well_id"] == "WELL_001"].set_index("date")["oil_bbl"]
yhat = dca.forecast(series, model="arps", kind="hyperbolic", horizon=12)
dca.plot(series, yhat)
