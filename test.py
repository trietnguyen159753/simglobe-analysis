import polars as pl

df = pl.scan_parquet("./data/constant.parquet")
print(
    df.select(
        "inflation",
        "real_gdp_growth",
        "unemployment",
        "budget_balance",
        "approval_index",
    ).describe()
)
