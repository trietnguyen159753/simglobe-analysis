import polars as pl
from .config import config
from .process import (
    load_process,
    eda_process,
    filter_process,
    regression_process,
    visualize_process,
)


def main():
    df = load_process(config.scenario)

    df_filter = filter_process(df, config.output_var)
    df_filter.collect().write_parquet("./cache/filter.parquet")

    df_filter = pl.scan_parquet("./cache/filter.parquet")

    eda_process(df_filter)

    df_regression = regression_process(df_filter)
    df_regression.collect().write_parquet("./cache/regression.parquet")

    df_regression = pl.scan_parquet("./cache/regression.parquet")

    print(df_regression.collect_schema().names())
    visualize_process(df_regression)


if __name__ == "__main__":
    main()
